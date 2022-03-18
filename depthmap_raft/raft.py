import sys
sys.path.append('RAFT-Stereo\core')
sys.path.append('RAFT-Stereo')

import argparse
import glob
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from raft_stereo import RAFTStereo
from utils.utils import InputPadder
from PIL import Image
from matplotlib import pyplot as plt


DEVICE = 'cuda'

class RAFTDepthEstimation(object):
    def __init__(self,args):
        self.args = args
        self.model = torch.nn.DataParallel(RAFTStereo(args), device_ids=[0])
        self.model.load_state_dict(torch.load(args.restore_ckpt))

        self.model = self.model.module
        self.model.to(DEVICE)
        self.model.eval()
    @torch.no_grad()
    def run(self, image1, image2):
        image1 = self.to_tensor(image1)
        image2 = self.to_tensor(image2)

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        _, flow_up = self.model(image1, image2, iters=self.args.valid_iters, test_mode=True)
        disparity = -flow_up.cpu().numpy().squeeze()

        return disparity

    def to_tensor(self, imgnumpy):
        img = torch.from_numpy(imgnumpy).permute(2, 0, 1).float()
        return img[None].to(DEVICE)