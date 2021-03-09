#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .base_options import BaseOptions

class TestOptions(BaseOptions):
	def initialize(self):
		BaseOptions.initialize(self)

		#model arguments
		self.mode = "test"
		self.isTrain = False
		self.enable_data_augmentation = False
		self.enable_cropping = True
