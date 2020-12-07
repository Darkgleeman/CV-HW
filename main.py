import glob

def pre_processor():
    train_imgs = glob.glob("dataset/train/img/*.jpg")
    train_density = glob.glob("dataset/train/density_maps/*.jpg")

