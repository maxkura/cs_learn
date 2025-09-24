import cv2
import numpy as np

picture_path= '/home/maxkura/exercise/cv_study/picture.png'
cat_path='/home/maxkura/exercise/cv_study/maodie.jpg'

pict= cv2.imread(picture_path)
cat= cv2.imread(cat_path)

#图片加载检测
if pict is None or cat is None:

    print("Error: Could not read the   image .")
    exit(1)

else:
    print(f'the shape of pict:{pict.shape}')
    print(f'the shape of cat:{cat.shape}')


#test
print(cat[1,10])

mask=np.zeros(pict.shape,dtype='uint8')
print('the shape of mask:')
print(mask.shape)

#尺寸获取与掩码构建
y1,x1=cat.shape[0:2]
y2,x2=pict.shape[0:2]
print('x1,x2,y1,y2')
print(x1,x2,y1,y2)

xl=int((x2-x1)/2)
xr=xl+x1

yl=int((y2-y1)/2)
yr=yl+y1

print('xl,xr,yl,yr')
print(xl,xr,yl,yr)

mask[yl:yr,xl:xr]=cat
print(mask.dtype,pict.dtype)
cv2.imwrite('/home/maxkura/exercise/cv_study/mask.png',mask)
#图片融合
result=cv2.addWeighted(pict, 0.5, mask, 1.2,-5)
cv2.imwrite('/home/maxkura/exercise/cv_study/result.png',result)
cv2.imwrite('/home/maxkura/exercise/cv_study/pict.png',pict)
cv2.imwrite('/home/maxkura/exercise/cv_study/cat.png',cat)







           