#!/usr/bin/env python
# -*- coding: utf-8 -*-
import gensim
import logging
import pandas

import numpy
import pickle
import random
import subprocess
from ast import literal_eval
from copy import deepcopy
from datetime import datetime

import scipy
import sklearn

import configargparse as argparse
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import os

from deepreg.datasets.calls_dataset import Datasets
from deepreg.models.models import Models, Loss
from deepreg.tools.log import setup_logging
from deepreg.tools.trainer import Trainer

parser = argparse.ArgumentParser(description='Training')

parser.add_argument('-c', '--config', is_config_file=True, help='config file path')

parser.add_argument('--results_dir', default='./results', help='results dir')
parser.add_argument('--experiment_dir', default='', help='Name for this experiment (will be a timestamp if empty)')

parser.add_argument('--dataset', )
parser.add_argument('--dataset_dir', help='dataset dir')
parser.add_argument('--workers', default=8, type=int, help='number of data loading workers (default: 8)')
parser.add_argument('--train_data_config', default="{}", help='data configuration')
parser.add_argument('--val_data_config', default="{}", help='data configuration')
parser.add_argument('--test_data_config', default="{}", help='data configuration')

parser.add_argument('--model', )
parser.add_argument('--model_config', default="{}", help='Architecture configuration')
parser.add_argument('--devices', default=0,)

parser.add_argument('--seed', default=0, type=int, help='Random seed')
parser.add_argument('--no_cuda', action='store_true', default=False, help='Train model')

parser.add_argument('--checkpoint', type=str, help='Checkpoint')
parser.add_argument('--log_text_and_att', action='store_true', help='Log outputs')
parser.add_argument('--log_result_file', type=str, help='Log results to file')
parser.add_argument('--embedding_vocab', type=str, help='Embedding lookup vocab Log outputs')
parser.add_argument('--eval_on_test', action='store_true', help='Eval on test')


def main(args, state_dict, orig_args):

    time_stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    print(args)

    root_logger = setup_logging()

    # Normalize the devices configuration
    if isinstance(args.devices, str):
        args.devices = literal_eval(args.devices)
    if isinstance(args.devices, int):
        args.devices = (args.devices,)

    # Assume the first device in the list is the main device
    if not args.no_cuda:
        main_gpu = 0
        if isinstance(args.devices, tuple):
            main_gpu = args.devices[0]
        if isinstance(args.devices, dict):
            main_gpu = args.devices.get('input', 0)
        torch.cuda.set_device(main_gpu)
        cudnn.benchmark = True


    if orig_args.seed > 0:
        random.seed(orig_args.seed)
        torch.manual_seed(orig_args.seed)

    #
    # Construct the dataset
    #


    if orig_args.dataset_dir is not None:
        dataset_dir = orig_args.dataset_dir
    else:
        dataset_dir = args.dataset_dir

    dataset = getattr(Datasets, args.dataset)
    train_data = dataset(dataset_dir=dataset_dir, **args.train_data_config, **args.experiment_settings)
    main_settings = deepcopy(args.train_data_config)
    for k,v in args.val_data_config.items():
        main_settings.pop(k)

    if orig_args.eval_on_test:
        args.test_data_config = literal_eval(args.test_data_config)
        val_data = dataset(dataset_dir=dataset_dir, **main_settings, **args.test_data_config, **args.experiment_settings)
    else:
        val_data = dataset(dataset_dir=dataset_dir, **main_settings, **args.val_data_config, **args.experiment_settings)

    model_config = args.model_config
    model_config['train_dataset'] = train_data.get_pickable_self()

    # Now initialize the model

    loss = getattr(Loss, args.loss)
    model_with_loss = loss(getattr(Models, args.model))
    model_with_loss.init_model(model_config)
    model_with_loss.model.load_state_dict(state_dict)
    logging.info(model_with_loss)

    if hasattr(model_with_loss.model, 'log_text_and_att') and orig_args.log_text_and_att:
        log_text_and_att_out = 'prediction-attention-{}-{}'.format(os.path.basename(args.config), time_stamp)
        model_with_loss.model.log_text_and_att = open(log_text_and_att_out, 'w')
    if hasattr(model_with_loss.model, 'embedding_vocab'):
        model_with_loss.model.embedding_vocab = gensim.models.fasttext.FastText.load('data/embeddings/fasttext_100').wv.index2word

    #
    # Define data loaders
    #

    val_loader = val_data.get_loader(
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    trainer_options = dict(
        criterion=None,
        cuda=not args.no_cuda,
        model_with_loss=model_with_loss,
        grad_clip=args.grad_clip,
        save_info={'config': args},
        devices=args.devices,
        print_freq=args.print_freq,
        save_freq=args.save_freq,
        save_epoch_freq=args.save_epoch_freq,
        eval_freq=args.eval_freq,
        model_select_metric=args.model_select_metric,
        evaluation_mode=True,
    )

    trainer = Trainer(**trainer_options)

    num_parameters = sum([l.nelement() for l in model_with_loss.model.parameters()])
    logging.info("number of parameters: %d", num_parameters)

    if not orig_args.no_cuda:
        # model.type(args.type)
        model_with_loss.model.cuda(main_gpu)
    else:
        import mkl
        mkl.set_num_threads(max(1,os.cpu_count() - 4))

    results = trainer.evaluate(val_loader)

    if hasattr(model_with_loss.model, 'phrases_pickle_dict') and orig_args.log_text_and_att:
        print(len(model_with_loss.model.phrases_pickle_dict))
        data = list()
        for k,v in model_with_loss.model.phrases_pickle_dict.items():
            v = numpy.array(v)
            if len(v) > 1:
                data.append((k, len(v), v.mean(), v.std(), v.max(),))
        df = pandas.DataFrame(data, columns="phrase,len,mean,std,max".split(','))
        df.sort_values(by='max', ascending=False).iloc[0:10000].to_csv('phrases_max_high_volatile-{}-{}.csv'.format(os.path.basename(args.config), time_stamp))
        df.sort_values(by='max', ascending=True).iloc[0:10000].to_csv('phrases_max_low_volatile-{}-{}.csv'.format(os.path.basename(args.config), time_stamp))
        df.sort_values(by='mean', ascending=False).iloc[0:10000].to_csv('phrases_mean_high_volatile-{}-{}.csv'.format(os.path.basename(args.config), time_stamp))
        df.sort_values(by='mean', ascending=True).iloc[0:10000].to_csv('phrases_mean_low_volatile-{}-{}.csv'.format(os.path.basename(args.config), time_stamp))
        df.sort_values(by='std', ascending=False).iloc[0:10000].to_csv('phrases_std_high_volatile-{}-{}.csv'.format(os.path.basename(args.config), time_stamp))
        df.sort_values(by='std', ascending=True).iloc[0:10000].to_csv('phrases_std_low_volatile-{}-{}.csv'.format(os.path.basename(args.config), time_stamp))
        # with open(phrases_pickle_dict, 'w') as f:
        #     f.write("phrase,mean,std,max")
        #     for k,v in model_with_loss.model.phrases_pickle_dict.items():
        #         v = numpy.array(v)
        #         f.write("{},{},{},{}".format(k, v.mean(), v.std(), v.max(),))

    for handler in root_logger.handlers: handler.flush()

    if orig_args.log_result_file:

        df = pandas.DataFrame(data=[[results[k] for k in sorted(results.keys())]], columns=list(sorted(results.keys())))
        if os.path.exists(orig_args.log_result_file):
            df.to_csv(orig_args.log_result_file, mode='a', header=False)
        else:
            df.to_csv(orig_args.log_result_file,)

if __name__ == '__main__':
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location={'cuda:0': 'cpu', 'cuda:1': 'cpu', 'cuda:2': 'cpu', 'cuda:3': 'cpu', })
    loaded_args = checkpoint['config']
    main(loaded_args, checkpoint['state_dict'], args)
