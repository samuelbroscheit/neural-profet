# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import copy
import numpy

import os

import pickle
import torch

from torch.utils.data import Dataset


class CallsDataset_V2(object):
    """docstring for Dataset."""

    def __init__(self,
                 dataset_dir,
                 data_padded_dat,
                 data_padded_loc,
                 data_padded_stats,
                 len_devices=1,
                 financial_features_mask=None,  # ['vix', 'size', 'volaprior', 'btm', 'sue', 'industry' , 'string']
                 ):
        super().__init__()

        self.dataset_dir = dataset_dir
        self.data_loc_list = None

        with open(os.path.join(dataset_dir, data_padded_loc), 'r') as f_loc:
            self.data_loc_list = list(map(int, f_loc.readlines()))

        self.data_stats = None
        with open(os.path.join(dataset_dir, data_padded_stats), 'rb') as f:
            self.data_stats = pickle.load(f)

        self.data_f = open(os.path.join(dataset_dir, data_padded_dat), 'rb')

        self.data_padded_masks = {
            'vix':       torch.FloatTensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            'size':      torch.FloatTensor([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            'volaprior': torch.FloatTensor([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            'btm':       torch.FloatTensor([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            'sue':       torch.FloatTensor([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            'industry':  torch.FloatTensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]),
            'string':    torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]),
        }

        if financial_features_mask is not None:
            if not isinstance(financial_features_mask, list):
                financial_features_mask = eval(financial_features_mask)
            self.mask = torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            for part in financial_features_mask:
                self.mask += self.data_padded_masks[part]
        else:
            raise Exception("Set mask either to [] or to [\'vix\', \'size\', \'volaprior\', \'btm\', \'sue\', \'industry\'] (or anything in between, cannot be None/unset.")

    def get_pickable_self(self):
        result = copy.copy(self)
        result.data_f = None
        result.data_loc_list = None
        return result

    def select_range(self, start, end):
        new_dataset = copy(self)
        new_dataset.items = new_dataset.items[start:end]
        return new_dataset

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[idx] for idx in range(index.start or 0, index.stop or len(self), index.step or 1)]
        self.data_f.seek(self.data_loc_list[index])
        indexed_call_id, \
        vola_after, \
        features, \
        presentation_toks_np, \
        question_1_toks_np, \
        answer_1_toks_np \
            = pickle.load(self.data_f)[:6]

        features = torch.FloatTensor(numpy.array(features).tolist())
        if self.mask is not None:
            features = features*self.mask

        # return  torch.log(1+torch.FloatTensor([vola_after])), \
        return  torch.FloatTensor([vola_after]), \
                features, \
                torch.LongTensor(presentation_toks_np), \
                torch.LongTensor(question_1_toks_np), \
                torch.LongTensor(answer_1_toks_np),

    def __len__(self):
        return len(self.data_loc_list)

    def get_loader(self,
                   batch_size=1,
                   shuffle=False,
                   sampler=None,
                   num_workers=0,
                   pin_memory=False,
                   drop_last=False):
        return torch.utils.data.DataLoader(self,
                                           batch_size=batch_size,
                                           sampler=sampler,
                                           shuffle=shuffle,
                                           num_workers=num_workers,
                                           pin_memory=pin_memory,
                                           drop_last=drop_last)


class Datasets:
    CallsDataset_V2 = CallsDataset_V2

if __name__ == '__main__':

    pass

    # with open(input_padded_data, 'rb') as f:
    #     line = pickle.load(f)
    #     indexed_call_id, \
    #     vola_after, \
    #     features, \
    #     presentation_toks_np, \
    #     question_1_toks_np, \
    #     answer_1_toks_np, \
    #     question_2_toks_np, \
    #     answer_2_toks_np = line
    #     print(line)

    # calls_dataset = CallsDataset(split=0)
    # calls_dataset_loader = calls_dataset.get_loader(
    #     batch_size=16,
    #     shuffle=True,
    #     num_workers=0,
    #     pin_memory=False,
    # )
    #
    # for bid, batch in enumerate(calls_dataset_loader):
    #     print(bid, batch[1].size())