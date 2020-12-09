from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import scipy
import scipy.ndimage
import scipy.io as scio
import skimage.io as io


def Global_Gaussain(shape,label):
	density=np.zeros(shape)
	n=label.shape[0]
	label_2=np.tile(np.sum(np.power(label,2),axis=1),(n,1))
	distance=label_2+label_2.T-2*np.dot(label,label.T)
	distance[distance<0]=0
	distance=np.sqrt(distance)
	distance_avg=np.sum(distance)/(n*(n-1))
	beta=0.3
	sigma=beta*distance_avg
	pixel=np.zeros(shape)
	for i in label.astype(np.uint16):
		pixel[i[0],i[1]]+=1
	pixel=scipy.ndimage.filters.gaussian_filter(pixel,sigma,mode='constant')
	density+=pixel
	return density

def Geometry_Gaussain(shape,label):
	density=np.zeros(shape)
	n=label.shape[0]
	label_2=np.tile(np.sum(np.power(label,2),axis=1),(n,1))
	distance=label_2+label_2.T-2*np.dot(label,label.T)
	distance[distance<0]=0
	distance=np.sqrt(distance)
	distance=np.sort(distance,axis=1)
	k=min(10,label.shape[0])
	beta=0.3
	sigma=np.mean(distance[:,:k],axis=1)*beta
	label[label<0]=0
	label=label.astype(np.uint16)
	for i in range(n):
		pixel=np.zeros(shape)
		pixel[label[i,0],label[i,1]]=1
		pixel=scipy.ndimage.filters.gaussian_filter(pixel,sigma[i],mode='constant')
		density+=pixel
	return density




path="dataset\\shanghaitech\\ShanghaiTech_Crowd_Counting_Dataset\\ShanghaiTech_Crowd_Counting_Dataset\\part_A_final\\train_data\\ground_truth\\"
path_image="dataset\\shanghaitech\\ShanghaiTech_Crowd_Counting_Dataset\\ShanghaiTech_Crowd_Counting_Dataset\\part_A_final\\train_data\\images\\"

#path="dataset\\shanghaitech\\ShanghaiTech_Crowd_Counting_Dataset\\ShanghaiTech_Crowd_Counting_Dataset\\part_B_final\\train_data\\ground_truth\\"
#path_image="dataset\\shanghaitech\\ShanghaiTech_Crowd_Counting_Dataset\\ShanghaiTech_Crowd_Counting_Dataset\\part_B_final\\train_data\\images\\"
savepath="data_preprocessed\\partB\\train\\"
for i in range(54,55):
	ground_truth_path=path+"GT_IMG_"+str(i)+".mat"
	ground_truth=scio.loadmat(ground_truth_path)
	ground_truth=ground_truth['image_info'][0,0][0,0][0]
	shape=Image.open(path_image+"IMG_"+str(i)+".jpg").size
	densitymap=Geometry_Gaussain(shape,ground_truth)
	np.save(savepath+"numpyarray\\"+str(i)+".npy",densitymap)
	print(i)
	print(ground_truth.shape[0])
	print(np.sum(densitymap))
	densitymap*=250/np.max(densitymap)
	img=densitymap.T.astype(np.uint8)
	'''
	img.save(savepath+"image\\"+str(i)+".jpg")
	'''
	io.imsave(savepath+"image\\"+str(11)+".jpg",img)
	plt.figure(0)
	plt.imshow(img, cmap = 'jet')
	plt.show()








