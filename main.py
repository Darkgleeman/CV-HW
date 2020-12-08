import glob
import torch
import cv2

def get_paths(directory):
    train_imgs = glob.glob("dataset/train/img/*.jpg")
    train_density = glob.glob("dataset/train/density_maps/*.jpg")
    return (train_imgs,train_density)

def pre_processor(img_path):
    raw_img = cv2.imread(img_path)

if __name__ == "__main__":
    pass