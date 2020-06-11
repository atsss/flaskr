import numpy as np
import cv2
import matplotlib.pyplot as plt

def kuwahara(pic,r=5,resize=False,rate=0.5):
    h,w,_=pic.shape
    if resize:pic=cv2.resize(pic,(int(w*rate),int(h*rate)));h,w,_=pic.shape
    pic=np.pad(pic,((r,r),(r,r),(0,0)),"edge")
    ave,var=cv2.integral2(pic)
    ave=(ave[:-r-1,:-r-1]+ave[r+1:,r+1:]-ave[r+1:,:-r-1]-ave[:-r-1,r+1:])/(r+1)**2
    var=((var[:-r-1,:-r-1]+var[r+1:,r+1:]-var[r+1:,:-r-1]-var[:-r-1,r+1:])/(r+1)**2-ave**2).sum(axis=2)

    def filt(i,j):
        return np.array([ave[i,j],ave[i+r,j],ave[i,j+r],ave[i+r,j+r]])[(np.array([var[i,j],var[i+r,j],var[i,j+r],var[i+r,j+r]]).argmin(axis=0).flatten(),j.flatten(),i.flatten())].reshape(w,h,_).transpose(1,0,2)

    filtered_pic = filt(*np.meshgrid(np.arange(h),np.arange(w))).astype(pic.dtype)

    return filtered_pic

pic=np.array(plt.imread("input.png"))
filtered_pic=kuwahara(pic,1,True,0.4)
plt.imshow(filtered_pic)
plt.show()
