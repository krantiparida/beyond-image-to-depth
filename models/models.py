import torch
import torchvision
import torch.nn as nn
from collections import OrderedDict
from .networks import RGBDepthNet, weights_init, \
    SimpleAudioDepthNet, attentionNet, MaterialPropertyNet

class ModelBuilder():
    # builder for audio stream
    def build_audiodepth(self, audio_shape=[2,257,121], weights=''):
        net = SimpleAudioDepthNet(8, audio_shape=audio_shape, audio_feature_length=512)
        net.apply(weights_init)
        if len(weights) > 0:
            print('Loading weights for audio stream')
            net.load_state_dict(torch.load(weights))
        return net

    #builder for visual stream
    def build_rgbdepth(self, ngf=64, input_nc=3, output_nc=1, weights=''):
        
        net = RGBDepthNet(ngf, input_nc, output_nc)

        net.apply(weights_init)
        if len(weights) > 0:
            print('Loading weights for visual stream')
            net.load_state_dict(torch.load(weights))
        return net

    def build_attention(self, weights=''):
        
        net = attentionNet(att_out_nc=512, input_nc=2*512)

        net.apply(weights_init)
        if len(weights) > 0:
            print('Loading weights for attention stream')
            net.load_state_dict(torch.load(weights))
        return net

    def build_material_property(self, nclass=10, weights=''):
        original_resnet = torchvision.models.resnet18(pretrained=False)
        net = MaterialPropertyNet(nclass, original_resnet)
        net.apply(weights_init)
        print('Training material property from scratch')
        
        if len(weights) > 0:
            print('Loading weights for material property stream')
            net.load_state_dict(torch.load(weights))
        return net
