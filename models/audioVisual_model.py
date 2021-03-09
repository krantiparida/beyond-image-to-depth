import os
import numpy as np
import torch
from torch import optim
import torch.nn.functional as F
from . import networks,criterion
from torch.autograd import Variable

class AudioVisualModel(torch.nn.Module):
    def name(self):
        return 'AudioVisualModel'

    def __init__(self, nets, opt):
        super(AudioVisualModel, self).__init__()
        self.opt = opt
        #initialize model
        self.net_rgbdepth, self.net_audio, self.net_attention, self.net_material = nets
        

    def forward(self, input, volatile=False):
        rgb_input = input['img']
        audio_input = input['audio']
        depth_gt = input['depth']

        audio_depth, audio_feat = self.net_audio(audio_input)
        img_depth, img_feat = self.net_rgbdepth(rgb_input)
        material_class, material_feat = self.net_material(rgb_input)
        audio_feat = audio_feat.repeat(1, 1, img_feat.shape[-2], img_feat.shape[-1]) #tile audio feature
        alpha, _ = self.net_attention(img_feat, audio_feat, material_feat)
        depth_prediction = ((alpha*audio_depth)+((1-alpha)*img_depth)) 

        
        output =  {'img_depth': img_depth * self.opt.max_depth,
                    'audio_depth': audio_depth * self.opt.max_depth,
                    'depth_predicted': depth_prediction * self.opt.max_depth, 
                    'attention': alpha,
                    'img': rgb_input,
                    'audio': audio_input,
                    'depth_gt': depth_gt}
        return output
