import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

#添加椒盐噪声
def add_salt_and_pepper_noise(image, salt_prob, pepper_prob):
    # 参数:
    # image: 输入图像（numpy数组）
    # salt_prob: 添加盐噪声的概率（0.0~1.0）
    # pepper_prob: 添加椒噪声的概率（0.0~1.0）

    noisy_image = np.copy(image)
    total_pixels = image.size
    num_salt = int(total_pixels * salt_prob)
    num_pepper = int(total_pixels * pepper_prob)

    # 添加盐噪声
    coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape]
    noisy_image[coords[0], coords[1]] = 255

    # 添加椒噪声
    coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape]
    noisy_image[coords[0], coords[1]] = 0

    return noisy_image

#最小值滤波函数
def min_filter(image, kernel_size):
    pad_size = kernel_size // 2
    padded_image = cv2.copyMakeBorder(image, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_REPLICATE)
    filtered_image = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded_image[i:i + kernel_size, j:j + kernel_size]
            filtered_image[i, j] = np.min(region)
    return filtered_image




#最大值滤波函数
def max_filter(image, kernel_size):
    pad_size = kernel_size // 2
    padded_image = cv2.copyMakeBorder(image, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_REPLICATE)
    filtered_image = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded_image[i:i + kernel_size, j:j + kernel_size]
            filtered_image[i, j] = np.max(region)

    return filtered_image


def main():
    pict=cv2.imread('C:/Users/ZhuanZ/code_cv/exp1/airplane.jpg',cv2.IMREAD_GRAYSCALE)


#问题一
    #生成高斯噪声
    gaussian_noise=np.random.normal(0, 16, pict.shape).astype(np.float32)
    print(gaussian_noise)
    pict_with_noise=cv2.add(pict.astype(np.float32),gaussian_noise).astype(np.uint8)
    #应用高斯滤波
    blurred_image = cv2.GaussianBlur(pict_with_noise, (5, 5), 0)


#问题二
    #定义卷积核
    kernel = np.array([[1, 0, 1],
                       [0, -4, 0],
                       [1, 0, 1]], dtype=np.float32)
    #进行滤波操作
    filtered_image = cv2.filter2D(pict, -1, kernel)


#问题三
    #添加椒盐噪声
    salt_pepper_noise_image = add_salt_and_pepper_noise(pict, 0.02, 0.02)
    #最小值滤波
    minfiltered_image = min_filter(salt_pepper_noise_image, 3)
    #最大值滤波
    maxfiltered_image = max_filter(salt_pepper_noise_image, 3)
    #中值滤波
    medBlur_image = cv2.medianBlur(salt_pepper_noise_image, 3)

    cv2.imshow('Original Image', pict)
    cv2.imshow('Noise Image', gaussian_noise)
    cv2.imshow('Image with Noise', pict_with_noise)
    cv2.imshow('Blurred Image', blurred_image)
    cv2.imshow('Filtered Image', filtered_image)
    cv2.imshow('Salt and Pepper Noise Image', salt_pepper_noise_image)
    cv2.imshow('Median Blurred Image', medBlur_image)
    cv2.imshow('Min Filtered Image', minfiltered_image)
    cv2.imshow('Max Filtered Image', maxfiltered_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()