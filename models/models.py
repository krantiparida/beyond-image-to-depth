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

    def build_material_property(self, nclass=10, weights='', init_weights=''):
        if len(init_weights) > 0:
            original_resnet = torchvision.models.resnet18(pretrained=True)
            net = MaterialPropertyNet(23, original_resnet)
            pre_trained_dict = torch.load(init_weights)['state_dict']
            pre_trained_mod_dict = OrderedDict()
            for k,v in pre_trained_dict.items():
                new_key = '.'.join(k.split('.')[1:])
                pre_trained_mod_dict[new_key] = v
            pre_trained_mod_dict = {k: v for k, v in pre_trained_mod_dict.items() if k in net.state_dict()}
            net.load_state_dict(pre_trained_mod_dict, strict=False)
            
            print('Initial Material Property Net Loaded')
            net.fc = nn.Linear(512, nclass)
        else:
            original_resnet = torchvision.models.resnet18(pretrained=False)
            net = MaterialPropertyNet(nclass, original_resnet)
            net.apply(weights_init)
            print('Moaterial Propert Net loaded')
        
        if len(weights) > 0:
            print('Loading weights for material property stream')
            net.load_state_dict(torch.load(weights))
        return net
