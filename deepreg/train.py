#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import logging
import random
import subprocess
from ast import literal_eval
from copy import deepcopy
from datetime import datetime

import configargparse as argparse
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import os

from deepreg.datasets.calls_dataset import Datasets
from deepreg.models.models import Models, Loss
from deepreg.tools.log import setup_logging
from deepreg.tools.misc import set_global_seeds
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
parser.add_argument('--loss', )
parser.add_argument('--devices', default=0,)
parser.add_argument('--type', default='torch.cuda.FloatTensor', help='Type of tensor - e.g torch.cuda.HalfTensor')
parser.add_argument('--batch_size', default=32, type=int, help='Mini-batch size (default: 32)')
parser.add_argument('--epochs', default=90, type=int, help='Number of total epochs to run')
parser.add_argument('--experiment_settings', default="{}", help='Experiment settings')
parser.add_argument('--start_epoch', default=0, type=int, help='Manual epoch number (useful on restarts)')

parser.add_argument('--optimization_config', default="{0: {'optimizer': SGD, 'lr':0.1, 'momentum':0.9}}", type=str, metavar='OPT', help='Optimization regime used')
parser.add_argument('--grad_clip', default='5.', type=str, help='Maximum grad norm value')

parser.add_argument('--print_freq', default=50, type=int, help='Print frequency (default: 10)')
parser.add_argument('--save_freq', default=1000, type=int, help='Save frequency (default: 10)')
parser.add_argument('--save_epoch_freq', default=1, type=int, help='Save frequency (default: 10)')
parser.add_argument('--eval_freq', default=2500, type=int, help='Evaluation frequency (default: 10)')
parser.add_argument('--model_select_metric', action='append', help='Evaluation metric to use for model selection [min_ppl|max_bleu|min_loss]')

parser.add_argument('--seed', default=0, type=int, help='Random seed')
parser.add_argument('--no_cuda', action='store_true', default=False, help='Train model')
parser.add_argument('--train', action='store_true', help='Train model')
parser.add_argument('--evaluate', action='store_true', help='Evaluate model')
parser.add_argument('--best_model', default='', type=str, help='Model name to evaluate')

parser.add_argument('--resume', default='', type=str, help='Path to checkpoint (default: none)')
parser.add_argument('--resume_filter', action="append", help='List of weight names that should be filtered out to resume')
parser.add_argument('--resume_freeze', action="append", help='Freeze the resumed parameters (either True for all or dict)')
parser.add_argument('--resume_pretraining', action='store_true', help='Use the checkpoint parameters as pretraining, else they are used as warm restart')

parser.add_argument('--copy_data_to_dev_shm', default=False, type=bool, help='Model name to evaluate')

parser.add_argument('--evaluate_config',  default='{}', type=str, help='Evaluation config')


def get_git_revision_hash():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'])


def get_git_revision_short_hash():
    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'])


def main(args, hyper_setting='', time_stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')):

    print(args)

    if args.experiment_dir is '':
        args.experiment_dir = '{}-{}-{}'.format(os.path.basename(args.config), hyper_setting, time_stamp)
    save_path = os.path.join(args.results_dir, args.experiment_dir)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    root_logger = setup_logging(os.path.join(save_path, 'log_%s.txt' % time_stamp))

    logging.info("saving to %s", save_path)
    logging.debug("run arguments: %s", args)

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

    # Set the seed
    if args.seed > 0:
        set_global_seeds(args.seed)

    #
    # Construct the dataset
    #

    dataset = getattr(Datasets, args.dataset)
    args.experiment_settings = literal_eval(args.experiment_settings)
    if isinstance(args.train_data_config, str):
        args.train_data_config = literal_eval(args.train_data_config)
    args.val_data_config = literal_eval(args.val_data_config)
    train_data = dataset(dataset_dir=args.dataset_dir, **args.train_data_config, **args.experiment_settings, len_devices=len(args.devices))
    main_settings = deepcopy(args.train_data_config)
    for k,v in args.val_data_config.items():
        main_settings.pop(k)
    val_data = dataset(dataset_dir=args.dataset_dir, **main_settings, **args.val_data_config, **args.experiment_settings, len_devices=len(args.devices))

    #
    # Optimization regime and model config
    #

    args.grad_clip = literal_eval(args.grad_clip)

    regime = args.optimization_config
    while type(regime) != list:
        regime = literal_eval(regime)

    if isinstance(args.model_config, str):
        model_config = literal_eval(args.model_config)
    else:
        model_config = args.model_config

    model_config['train_dataset'] = train_data.get_pickable_self()

    args.model_config = model_config

    # Now initialize the model

    model = getattr(Models, args.model)
    loss = getattr(Loss, args.loss)
    model_with_loss = loss(model)
    model_with_loss.init_model(model_config)
    logging.info(model_with_loss)

    #
    # Define data loaders
    #

    train_loader = train_data.get_loader(
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        # drop_last=True,
    )

    val_loader = val_data.get_loader(
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        # drop_last=True,
    )

    trainer = Trainer(
        criterion=None,
        cuda=not args.no_cuda,
        model_with_loss=model_with_loss,
        grad_clip=args.grad_clip,
        save_path=save_path,
        save_info={'config': args},
        regime=regime,
        devices=args.devices,
        print_freq=args.print_freq,
        save_freq=args.save_freq,
        save_epoch_freq=args.save_epoch_freq,
        eval_freq=args.eval_freq,
        model_select_metric=args.model_select_metric,
    )

    num_parameters = sum([l.nelement() for l in model_with_loss.model.parameters()])
    logging.info("number of parameters: %d", num_parameters)

    if not args.no_cuda:
        # model.type(args.type)
        model_with_loss.to(args.devices[0])
    else:
        import mkl
        mkl.set_num_threads(max(1,os.cpu_count() - 4))

    # optionally resume from a checkpoint
    if args.resume:
        checkpoint_file = args.resume
        if os.path.isdir(checkpoint_file):
            # results.load(os.path.join(checkpoint_file, 'results.csv'))
            checkpoint_file = os.path.join(checkpoint_file, 'model_best.pth.tar')
        if os.path.isfile(checkpoint_file):
            trainer.load(checkpoint_file,
                         resume_filter=args.resume_filter,
                         freeze_param=args.resume_freeze,
                         as_pretraining=args.resume_pretraining,
                         weight_map=None,
                         )
        else:
            logging.error("no checkpoint found at '%s'", args.resume)

    logging.info('training regime: %s', regime)
    trainer._epoch = args.start_epoch

    if args.train:
        try:
            while trainer.epoch < args.epochs:
                # train for one epoch
                trainer.run(train_loader, val_loader)
                if trainer.terminate:
                    break
        except KeyboardInterrupt:
            for handler in root_logger.handlers: handler.flush()
            pass

    for handler in root_logger.handlers: handler.flush()

    return trainer.valid_loss, trainer.average_valid_loss.avg, trainer.valid_sr, trainer.valid_kt, trainer.valid_pr, trainer.valid_rmse


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
