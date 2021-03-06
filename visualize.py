# -*- coding: utf-8 -*-
import skimage.io
import torch
import numpy as np
import matplotlib.pyplot as plt
import h5py

# show Chinese 
plt.rcParams['font.sans-serif'] = ['SimHei']

def density_show(densityImage):
    plt.figure()
    # no coodinate axis
    plt.axis('off')
    # the width of white margin of x and y direction
    plt.margins(0, 0)
    # candidate_color = ['CMRmap', 'YlGnBu_r', 'cubehelix', 'jet', 'terrain']
    # plt.imshow(densityImage, cmap = 'terrain')
    plt.imshow(densityImage, cmap = 'jet')
    # plt.savefig(name, dpi = 600, bbox_inches = 'tight', pad_inches = 0)
    plt.show()

def loss_show(running_loss ,file_name ,epoches = None):
    i = len(file_name) - 1
    while i > 0 and (file_name[i] != '\\' or file_name[i] != '/'):
        i -= 1
    if epoches == None:
        epoches = [i + 1 for i in range(len(running_loss))]
    plt.figure()
    plt.grid()
    plt.xlabel("epoch")
    plt.ylabel(file_name[i+1:-3])
    plt.plot(epoches, running_loss, linestyle = '-', color = 'c')
    plt.savefig(file_name)
    # plt.show()

def MAE_show(MAE_loss ,file_name):
    i = len(file_name) - 1
    while i > 0 and (file_name[i] != '\\' or file_name[i] != '/'):
        i -= 1
    if epoches == None:
        epoches = [i + 1 for i in range(len(MAE_loss))]
    plt.figure()
    plt.grid()
    plt.xlabel("frames")
    plt.ylabel(file_name[i+1:-3])
    plt.plot(epoches, MAE_loss, linestyle = '-', color = 'c')
    plt.savefig(file_name)

if __name__ == "__main__":
    # test of density_show

    f = h5py.File(r"C:\Users\12078\Documents\Tencent Files\1207820254\FileRecv\IMG_54.h5", 'r')
    for key in f.keys():
        print(f[key])
        density_show(f[key][()])


    # #########################################
    # # test of train loss function
    # model = build_model_test_XXXXXXXXXXXX
    # model_train(None, model)

    # ## 需要train时候的loss

    # #########################################
    # # test of test loss function

    # n = 0
    # mae = 0
    # mse = 0
    # for i in range(len(test_dataset)):
    #     img, density = test_dataset[i]
    #     # img = preprocess_input(img)
    #     pred = model.predict(img)
    #     pred_values = pred.sum()
    #     truth = density.sum()
    #     mae = mae + abs(truth - pred_values)
    #     n += 1
    #     mse += (truth - pred_values) * (truth - pred_values)
    # mae = mae / n
    # mse = np.sqrt(mse / n)
    # print(mae)
    # print(mse)
