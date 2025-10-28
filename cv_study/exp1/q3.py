import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
#参数
D0 = 50  # 截止频率
#快速傅里叶变换并中心化：返回复数频谱 (H, W, 2)
def fftshift(pict):
    pict_float = pict.astype(np.float32)
    pict_fft = cv2.dft(pict_float, flags=cv2.DFT_COMPLEX_OUTPUT)  # (H, W, 2)
    dft_shift = np.fft.fftshift(pict_fft, axes=(0, 1))
    return dft_shift


# 仅用于显示：将复数频谱转换为幅值 (H, W)
def spectrum_magnitude(fft_shifted):
    return cv2.magnitude(fft_shifted[:, :, 0], fft_shifted[:, :, 1])

#快速傅里叶变换的逆变换并中心化：输入复数频谱 (H, W, 2)，输出空间域图像 (H, W)
def ifftshift(fft_shifted):
    fft_unshifted = np.fft.ifftshift(fft_shifted, axes=(0, 1))
    # 输出实部，自动缩放
    img_back = cv2.idft(fft_unshifted, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)  # (H, W)
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
    return img_back.astype(np.uint8)

# 理想低通滤波器
def ideal_low_pass_filter(fft_shifted, D0):
    # 在复数频谱上进行滤波：H 为 (H, W)，通过广播乘到 (H, W, 2)
    h, w = fft_shifted.shape[:2]
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    D = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)
    H = (D <= D0).astype(np.float32)
    return fft_shifted * H[:, :, None]

#2阶巴特沃斯低通滤波器
def butterworth_low_pass_filter(fft_shifted, D0, order):
    h, w = fft_shifted.shape[:2]
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    D = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)
    H = 1.0 / (1.0 + (D / (D0 + 1e-6)) ** (2 * order))
    H = H.astype(np.float32)
    return fft_shifted * H[:, :, None]

#FFT幅值2D转化

def to2d(magnitude):
    magnitude_log = np.log1p(magnitude)  # log(1+x) 
    magnitude_normalized = cv2.normalize(magnitude_log, None, 0, 255, cv2.NORM_MINMAX)
    magnitude_display = np.uint8(magnitude_normalized)
    return magnitude_display

# 绘制3D频谱图
def plot_3d(magnitude):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = np.arange(np.log1p(magnitude).shape[1])
    y = np.arange(np.log1p(magnitude).shape[0])
    x, y = np.meshgrid(x, y)

    # 绘制3D表面
    surf = ax.plot_surface(
    x, y, np.log1p(magnitude),
    cmap='viridis',
    edgecolor='none',
    alpha=0.8,
    rstride=1,
    cstride=1
    )
    # 添加颜色条
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, pad=0.1)

    # 设置标签和标题
    ax.set_xlabel('Frequency (x)', fontsize=10)
    ax.set_ylabel('Frequency (y)', fontsize=10)
    ax.set_zlabel('Log Magnitude', fontsize=10)
    ax.set_title('3D Log Magnitude Spectrum (Centered)', fontsize=12)

    # 调整视角
    ax.view_init(elev=30, azim=45)

    # 优化布局
    plt.tight_layout()

    # 保存和显示结果
    plt.show()


# 创建5x5高斯空间滤波器,size: 滤波器尺寸 (必须是奇数),sigma: 高斯标准差
def create_gaussian_kernel(size=5, sigma=1.0):
    center = size // 2
    kernel = np.zeros((size, size))
    
    for x in range(size):
        for y in range(size):
            # 计算高斯函数
            dx = x - center
            dy = y - center
            kernel[x, y] = (1 / (2 * math.pi * sigma**2)) * math.exp(-(dx**2 + dy**2) / (2 * sigma**2))
    
    # 归一化
    kernel = kernel / np.sum(kernel)
    return kernel

# 创建对应的频域高斯滤波器,,size: 滤波器尺寸 (必须是奇数),sigma: 高斯标准差
def create_freq_domain_gaussian(size=5, sigma=1.0):
    # 创建空间域高斯核
    spatial_kernel = create_gaussian_kernel(size, sigma)
    
    # 计算频域滤波器 (空间域核的FFT)
    freq_kernel = np.fft.fft2(spatial_kernel, s=(size, size))
    freq_kernel = np.fft.fftshift(freq_kernel)
    
    # 归一化 (使中心值为1)
    freq_kernel = freq_kernel / np.max(freq_kernel)
    
    return freq_kernel

# 应用空间域高斯滤波,image: 输入图像 (灰度),kernel: 高斯滤波器
def apply_gaussian_filter(image, kernel):

    # 确保是浮点类型
    image_float = image.astype(np.float32)
    
    # 应用滤波
    filtered = cv2.filter2D(image_float, -1, kernel)
    
    # 归一化到0-255
    filtered = cv2.normalize(filtered, None, 0, 255, cv2.NORM_MINMAX)
    return filtered.astype(np.uint8)

# 应用频域高斯滤波,image: 输入图像 (灰度),freq_kernel: 频域高斯滤波器,image_size: 图像尺寸 (H, W)
def apply_freq_domain_gaussian(image, freq_kernel, image_size):

    # 转换为浮点类型
    image_float = image.astype(np.float32)
    
    # 计算FFT (中心化)
    fft_img = np.fft.fft2(image_float)
    fft_shifted = np.fft.fftshift(fft_img)
    
    # 创建与图像大小相同的频域滤波器 (用零填充)
    freq_filter = np.zeros(image_size, dtype=np.complex64)
    center = (image_size[0]//2, image_size[1]//2)
    
    # 将频域滤波器放在中心
    h, w = freq_kernel.shape
    start_h = center[0] - h//2
    start_w = center[1] - w//2
    freq_filter[start_h:start_h+h, start_w:start_w+w] = freq_kernel
    
    # 与频谱相乘
    filtered_fft = fft_shifted * freq_filter
    
    # 逆FFT
    filtered_shifted = np.fft.ifftshift(filtered_fft)
    filtered_img = np.fft.ifft2(filtered_shifted)
    
    # 取实部并归一化
    filtered_img = np.real(filtered_img)
    filtered_img = cv2.normalize(filtered_img, None, 0, 255, cv2.NORM_MINMAX)
    
    return filtered_img.astype(np.uint8)

# 绘制滤波器响应的3D和2D图像
def plot_filter_response(kernel, title, ax):
    # 3D图
    x = np.arange(kernel.shape[1])
    y = np.arange(kernel.shape[0])
    X, Y = np.meshgrid(x, y)
    
    ax.plot_surface(X, Y, kernel, cmap='viridis', edgecolor='none', alpha=0.8)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('x', fontsize=8)
    ax.set_ylabel('y', fontsize=8)
    ax.set_zlabel('Amplitude', fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

def main():
    pict=cv2.imread('C:/Users/ZhuanZ/code_cv/exp1/barbara.jpg',cv2.IMREAD_GRAYSCALE)

    # FFT 复数频谱
    fft = fftshift(pict)                     # (H, W, 2)
    magnitude = spectrum_magnitude(fft)      # (H, W)
    magnitude_display = to2d(magnitude)

    # 应用理想低通滤波器（在复数频谱上）
    fft_ILPF = ideal_low_pass_filter(fft, D0)
    image_ILPF = ifftshift(fft_ILPF)
    magnitude_ILPF = spectrum_magnitude(fft_ILPF)
    magnitude_ILPF_display = to2d(magnitude_ILPF)

    # 应用二阶巴特沃斯低通滤波器
    fft_BLPF = butterworth_low_pass_filter(fft, D0, order=2)
    image_BLPF = ifftshift(fft_BLPF)
    magnitude_BLPF = spectrum_magnitude(fft_BLPF)
    magnitude_BLPF_display = to2d(magnitude_BLPF)

    plot_3d(magnitude)
    plot_3d(magnitude_ILPF)
    plot_3d(magnitude_BLPF)

    Skernel=create_gaussian_kernel(size=5, sigma=1.0)
    Fkernel=create_freq_domain_gaussian(size=5, sigma=1.0)

    #应用高斯滤波器进行空间滤波
    spatial_filtered = apply_gaussian_filter(pict, Skernel)

    #应用高斯滤波器进行频域滤波
    freq_filtered = apply_freq_domain_gaussian(pict, Fkernel, pict.shape)

    # 显示结果
    cv2.imshow('Original Image', pict)
    cv2.imshow('Image after FFT', magnitude_display)
    cv2.imshow('After Ideal Low Pass Filter', magnitude_ILPF_display)
    cv2.imshow('After Butterworth Low Pass Filter', magnitude_BLPF_display)
    cv2.imshow('Image after Ideal Low Pass Filter', image_ILPF)
    cv2.imshow('Image after Butterworth Low Pass Filter', image_BLPF)
    cv2.imshow('Spatial Domain Gaussian Filtered', spatial_filtered)
    cv2.imshow('Frequency Domain Gaussian Filtered', freq_filtered)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()