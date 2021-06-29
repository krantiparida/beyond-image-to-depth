#!/usr/bin/env python

import argparse
import os
from util import util
import torch

class BaseOptions():
	def __init__(self):
		self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
		self.initialized = False

	def initialize(self):
		self.parser.add_argument('--img_path', type=str, default= '/data1/kranti/audio-visual-depth/dataset/visual_echoes/images/mp3d_split_wise', help='root path to the file that contains the .pkl file')
		self.parser.add_argument('--metadatapath', type=str, default= '/data1/kranti/audio-visual-depth/dataset/visual_echoes/metadata/mp3d', help= 'path to metadata file for different split')
		self.parser.add_argument('--audio_path', type=str, default= '/data1/kranti/audio-visual-depth/dataset/visual_echoes/echoes/mp3d/echoes_navigable', help='path to the folder that contains echo responses')
		self.parser.add_argument('--checkpoints_dir', type=str, default= '', help='path to save checkpoints')
		self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
		self.parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
		self.parser.add_argument('--nThreads', default=8, type=int, help='# threads for loading data')
		self.parser.add_argument('--audio_length', default=0.06, type=float, help='audio length, default 0.06s')
		self.parser.add_argument('--audio_normalize', type=bool, default=False, help='whether to normalize the audio')
		self.parser.add_argument('--image_transform', type=bool, default=True, help='whether to transform the image data')
		self.parser.add_argument('--image_resolution', default=128, type=int, help='the resolution of image for cropping')
		self.parser.add_argument('--dataset', default='mp3d', type=str, help='replica/mp3d')
		## scratch was added after one step of training done with material property initialization
		self.initialized = True

	def parse(self):
		if not self.initialized:
			self.initialize()
		self.opt = self.parser.parse_args()

		self.opt.mode = self.mode
		self.opt.isTrain = self.isTrain
		self.opt.enable_img_augmentation = self.enable_data_augmentation
		self.opt.enable_cropping = self.enable_cropping 
		
		
		# dataset specific paramters
		self.opt.scenes = {}
		if self.opt.dataset == 'mp3d':
			self.opt.audio_shape = [2,257,121]
			self.opt.audio_sampling_rate = 16000
			self.opt.max_depth = 10.0
			train_scenes_file = os.path.join(self.opt.metadatapath, 'mp3d_scenes_train.txt')
			val_scenes_file = os.path.join(self.opt.metadatapath, 'mp3d_scenes_val.txt')
			test_scenes_file = os.path.join(self.opt.metadatapath, 'mp3d_scenes_test.txt')
			with open(train_scenes_file) as f:
				content = f.readlines()
			self.opt.scenes['train'] = [x.strip() for x in content]

			with open(val_scenes_file) as f:
				content = f.readlines()
			self.opt.scenes['val'] = [x.strip() for x in content]

			with open(test_scenes_file) as f:
				content = f.readlines()
			self.opt.scenes['test'] = [x.strip() for x in content]
		
		if self.opt.dataset == 'replica':
			self.opt.audio_shape = [2,257,166]
			self.opt.audio_sampling_rate = 44100
			self.opt.max_depth = 14.104
			self.opt.scenes['train'] = ['apartment_0', 'apartment_1', 
			'frl_apartment_0', 'frl_apartment_1', 'frl_apartment_2', 'frl_apartment_3',
			'frl_apartment_4', 'office_0', 'office_1', 'office_2', 'office_3', 
			'hotel_0', 'room_0', 'room_1', 'room_2']
			self.opt.scenes['test'] = ['apartment_2', 'frl_apartment_5', 'office_4']
			

		str_ids = self.opt.gpu_ids.split(',')
		self.opt.gpu_ids = []
		for str_id in str_ids:
			id = int(str_id)
			if id >= 0:
				self.opt.gpu_ids.append(id)

		args = vars(self.opt)
		print('------------ Options -------------')
		for k, v in sorted(args.items()):
			print('%s: %s' % (str(k), str(v)))
		print('-------------- End ----------------')


		# save to the disk
		expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.dataset)
		self.opt.expr_dir = expr_dir
		util.mkdirs(expr_dir)
		file_name = os.path.join(expr_dir, 'opt.txt')
		with open(file_name, 'wt') as opt_file:
			opt_file.write('------------ Options -------------\n')
			for k, v in sorted(args.items()):
				opt_file.write('%s: %s\n' % (str(k), str(v)))
			opt_file.write('-------------- End ----------------\n')

		return self.opt
