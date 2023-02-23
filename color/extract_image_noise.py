import cv2
import os
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from models import ADNet
from utils import *
import matplotlib.pyplot as plt


print('Loading model ...\n')
net = ADNet(channels=3, num_of_layers=17)
device_ids = [0]
model = nn.DataParallel(net, device_ids=device_ids).cuda()
model.load_state_dict(torch.load('color/c75/model_70.pth'))
model.eval()

size = 500
x = 1000
y = 1000
image = cv2.cvtColor(cv2.imread("gonpachi.JPG"), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
image = image[y:size+y,x:size+x,:]
image = torch.tensor(image)
image = image.permute(2, 0, 1)[None,...].repeat(3,1,1,1)

with torch.no_grad():
    out = model(image)
    noise = image - out.cpu()

plt.subplot(1,3,1)
plt.imshow(image[0,...].permute(1,2,0).cpu().numpy())
plt.subplot(1,3,2)
plt.imshow(out[0,...].permute(1,2,0).cpu().numpy())
plt.subplot(1,3,3)
plt.imshow(noise[0,0,...].cpu().numpy())
plt.colorbar()
plt.show()
