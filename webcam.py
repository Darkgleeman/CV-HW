# coding: utf-8
import imageio
import numpy as np
from tqdm import tqdm
import yaml
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

import cv2

################### local module ######################
import CNN
from CNN import MCNN

######################################################
device = 'cuda'

def MAE_show(MAE_loss, file_name, epoches = None):
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

def main():
    batch_size = 1
    device = "cuda"

    net = MCNN().to(device)
    net_dict = torch.load('MSE_model_200_1e-05_20.ckpt', map_location=torch.device('cuda'))
    net.eval()
    net.load_state_dict(net_dict)
    sum_list = []
    # Given a camera
    # before run this line, make sure you have installed `imageio-ffmpeg`
    reader = imageio.get_reader("<video0>", input_params=['-framerate', '30'])

    # the simple implementation of average smoothing by looking ahead by n_next frames
    # assert the frames of the video >= n
    n_pre, n_next = 1, 1

    n = n_pre + n_next + 1

    fourcc = cv2.VideoWriter_fourcc('m','p','4','v')

    # run

    for i, frame in tqdm(enumerate(reader)):
        # frame_bgr = frame[..., ::-1]  # RGB->BGR
        frame_bgr = frame
        inputs = torch.from_numpy(frame_bgr)
        cv2.imwrite("./input.jpg", inputs.detach().cpu().numpy()[:,:,::-1])
        
        cv2.imshow("input", inputs.detach().cpu().numpy()[:,:,::-1])
        k = cv2.waitKey(20)
        if k & 0xff == 'q':
            break
        
        inputs = inputs.permute(2, 0, 1).type(dtype=torch.float).to(device)
        inputs=inputs.unsqueeze(0)

        outputs = net(inputs)
        sum = torch.sum(outputs[0][0]).item()*16
        sum_list.append(sum)
        print("sum", sum)
        outputs[0][0] *= (255 / torch.max(outputs[0][0]))

        cv2.imwrite("./output.jpg", outputs[0][0].detach().cpu().numpy())
        print(outputs[0][0].shape)
        print(outputs.shape)
        res = outputs[0].permute(1, 2, 0)
        print(res.shape)
        cv2.imshow("output", res.detach().cpu().numpy()/255)
        k = cv2.waitKey(20)
        if k & 0xff == 'q':
            break
        # MAE_show(sum_list, './video_MAE.jpg')


if __name__ == '__main__':
    main()

