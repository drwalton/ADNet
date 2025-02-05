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

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="ADNet_Test")
parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")
parser.add_argument("--logdir", type=str, default="color/c75", help='path of log files')
parser.add_argument("--test_data", type=str, default="color/data/McMaster", help='test on Set12 or Set68')
parser.add_argument("--test_noiseL", type=float, default=15, help='noise level used on test set')
opt = parser.parse_args()

def normalize(data):
    return data/255.

def main():
    # Build model
    print('Loading model ...\n')
    net = ADNet(channels=3, num_of_layers=opt.num_of_layers)
    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    model.load_state_dict(torch.load(os.path.join(opt.logdir, 'model_70.pth')))
    model.eval()
    # load data info
    print('Loading data info ...\n')
    files_source = glob.glob(os.path.join(opt.test_data, '*'))
    files_source.sort()
    # process data
    psnr_test = 0
    for f in files_source:
        # image
        Img = cv2.imread(f)
        Img = torch.tensor(Img)
        #print Img.shape
        Img = Img.permute(2,0,1)
        Img = Img.numpy()
        a1, a2, a3 = Img.shape
        Img = np.tile(Img,(3,1,1,1))  #expand the dimensional 
        Img = np.float32(normalize(Img))
        ISource = torch.Tensor(Img)
        # noise
        torch.manual_seed(12) 
        noise = torch.FloatTensor(ISource.size()).normal_(mean=0, std=opt.test_noiseL/255.)
        # noisy image
        INoisy = ISource + noise
        ISource = Variable(ISource) 
        INoisy = Variable(INoisy) 
        ISource= ISource.cuda() 
        INoisy = INoisy.cuda() 
        with torch.no_grad(): # this can save much memory
            Out = torch.clamp(model(INoisy), 0., 1.)
        ## if you are using older version of PyTorch, torch.no_grad() may not be supported
        # ISource, INoisy = Variable(ISource.cuda(),volatile=True), Variable(INoisy.cuda(),volatile=True)
        # Out = torch.clamp(INoisy-model(INoisy), 0., 1.)
        psnr = batch_PSNR(Out, ISource, 1.)
        psnr_test += psnr
        print("%s PSNR %f" % (f, psnr))
    psnr_test /= len(files_source)
    print("\nPSNR on test data %f" % psnr_test)

if __name__ == "__main__":
    main()
