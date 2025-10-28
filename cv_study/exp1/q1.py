import cv2
import numpy as np
import matplotlib.pyplot as plt

# 计算并显示图像的归一化像素强度统计直方图
def pict_normalized(pict):
    y=pict.shape[0]
    x=pict.shape[1]
    stats=np.zeros((256,),dtype=int)
    for i in range(x):
        for j in range(y):

            stats[pict[j,i]]+=1
    stats=stats/(x*y)
    plt.bar(range(256),stats)
    plt.show()
    return stats


#直方图均衡化
def pict_processed(pict):
    y=pict.shape[0]
    x=pict.shape[1]
    stats=np.zeros((256,),dtype=int)
    for i in range(x):
        for j in range(y):
            stats[pict[j,i]]+=1
    cdf=np.zeros((256,),dtype=float)
    cdf[0]=stats[0]
    for i in range(1,256):
        cdf[i]=cdf[i-1]+stats[i]
    cdf=cdf/(x*y)
    pict_new=np.zeros((y,x),dtype=np.uint8)
    for i in range(x):
        for j in range(y):
            pict_new[j,i]=np.uint8(cdf[pict[j,i]]*255)
    return pict_new

def main():
    pict=cv2.imread('C:/Users/ZhuanZ/code_cv/exp1/Low illumination.jpg',cv2.IMREAD_GRAYSCALE)   
    print('Original Image Shape:', pict.shape)
    #print('Original Image :', pict)
    stat=pict_normalized(pict)
    pict_after=pict_processed(pict)
    stat_after=pict_normalized(pict_after)
    # print('Pixel Intensity Statistics:', stats)

    cv2.imshow('Original Image', pict)
    cv2.imshow('Processed Image', pict_after)
    # 等待按键输入
    cv2.waitKey(0)
    # 关闭所有窗口
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()