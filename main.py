import glob
import torch
import cv2

def get_paths(directory):
    train_imgs = glob.glob("dataset/train/img/*.jpg")
    train_density = glob.glob("dataset/train/density_maps/*.jpg")
    return (train_imgs,train_density)

def pre_processor(img_path,density_path):
    raw_img = cv2.imread(img_path)#size:1024 by 768 
    raw_density = cv2.imread(density_path)
    splited_img = [[],[],[]]
    splited_density = [[],[],[]]
    h = 512
    w = 384
    for i in range(3):
        for j in range(3):
            splited_img[i].append(raw_img[h/2*j:h/2*(j+2),w/2*i:w/2*(i+2)]) #split into 9 parts
            splited_density[i].append(raw_density[h/2*j:h/2*(j+2),w/2*i:w/2*(i+2)])

if __name__ == "__main__":
    pass