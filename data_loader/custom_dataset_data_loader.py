#!/usr/bin/env python

import torch.utils.data

def CreateDataset(opt):
    dataset = None
    from data_loader.audio_visual_dataset import AudioVisualDataset
    dataset = AudioVisualDataset()
    dataset.initialize(opt)
    return dataset

class CustomDatasetDataLoader():
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        self.dataset = CreateDataset(opt)
        shuff = False
        if opt.mode == "train":
            print('Shuffling the dataset....')
            shuff= True
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=shuff,
            num_workers=int(opt.nThreads))

    def load_data(self):
        return self

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            yield data
