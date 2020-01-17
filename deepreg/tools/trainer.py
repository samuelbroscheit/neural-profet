import math
import numpy
import os
import re
import sklearn.metrics

import scipy.stats
import time
import logging
from typing import Union

import torch
import torch.nn as nn
from itertools import cycle
from torch.nn.parallel import DataParallel
from torch.nn.utils import clip_grad_norm
import shutil

from deepreg.models.models import ModelWrappedWithMSELoss
from deepreg.tools.log import ResultsLog
from deepreg.tools.meters import AverageMeter, ScipyStatsMeter, SkLearnMetricMeter
from deepreg.tools.optim import OptimRegime

def running_mean(new, old=None, momentum=0.9):
    if old is None:
        return new
    else:
        return momentum*old + (1-momentum)*new

class Trainer(object):
    """class for Trainer.

     regime is an ordered list by epochs
     (can be a float indicating relative progress)"""

    def __init__(self,
                 model_with_loss:ModelWrappedWithMSELoss=None,
                 regime=None,
                 criterion=None,
                 print_freq=10,
                 eval_freq=1000,
                 save_freq=1000,
                 save_epoch_freq=1,
                 grad_clip=None,
                 save_info={},
                 save_path='.',
                 checkpoint_filename='checkpoint%s.pth.tar',
                 keep_checkpoints=5,
                 devices=None,
                 model_select_metric=['min_loss'],
                 cuda=True,
                 do_filter_for_weight_decay=True,
                 evaluation_mode=False,
                 patience_epochs=10,
                 patience_criterion_treshold=100000,
                 patience_criterion_change=0.001,
                 ):
        super(Trainer, self).__init__()

        self.model, self.criterion = model_with_loss.model, model_with_loss.criterion
        self.model_with_loss = model_with_loss
        self.cuda = cuda

        if not evaluation_mode:
            self.regime = regime
            self.optimizers = list()
            if do_filter_for_weight_decay:
                for filter_for_weight_decay in [1, 0]:
                    named_params = self.parameters(model_with_loss, do_filter=True, filter_for_weight_decay=filter_for_weight_decay)
                    named_params = filter(lambda x: x[1].requires_grad, named_params)
                    list_named_params = list(named_params)
                    if len(list_named_params ) > 0:
                        logging.info("Parameters with{} weight decay:".format('out' if filter_for_weight_decay else ''))
                        params = list()
                        for name, param in list_named_params:
                            logging.info(name)
                            params.append(param)
                        self.optimizers.append(OptimRegime(params, regime=regime, filter_weight_decay=filter_for_weight_decay))
            self.grad_clip = grad_clip

        self.save_info = save_info
        torch.set_num_threads(4)
        print('Using CUDA: {} || Number of CPU cores for torch {}'.format(self.cuda, torch.get_num_threads()))
        self.print_freq = print_freq
        self.eval_freq = eval_freq

        self.devices = devices

        if isinstance(self.devices, tuple) and len(self.devices) > 1:
            self.model_with_loss = DataParallel(self.model_with_loss, self.devices)
        self.save_path = save_path
        self.save_freq = save_freq
        self.model_select_metric = model_select_metric
        self.checkpoint_filename = checkpoint_filename

        self.keep_checkpoints = keep_checkpoints

        self.results = ResultsLog(os.path.join(save_path, 'results.%s')% 'csv', os.path.join(save_path, 'results.%s')% 'html')

        self._epoch = 0
        self.training_steps = 0

        self.counter = cycle(range(self.keep_checkpoints))
        self.save_epoch_freq_cycle = cycle(range(save_epoch_freq))
        self.save_epoch_freq = -1

        self.terminate = False
        self.patience_epochs = patience_epochs
        self.terminate_epochs = -1
        self.patience_criterion_treshold = patience_criterion_treshold

        self.moving_average_criterion_change = None
        self.patience_criterion_change = patience_criterion_change
        self.valid_loss = float('inf')
        self.valid_sr = -float('inf')
        self.valid_kt = -float('inf')
        self.valid_pr = -float('inf')
        self.valid_rmse = float('inf')
        self.average_valid_loss = AverageMeter()
        self.last_val_loss = None

    @property
    def epoch(self):
        return math.floor(self._epoch)

    def parameters(self, parent_module, memo=None, prefix='', do_filter=False, filter_for_weight_decay=None, ):
        for name, param in self.named_parameters(parent_module, memo, prefix, do_filter, filter_for_weight_decay):
            yield name, param

    def named_parameters(self, parent_module, memo=None, prefix='', do_filter=False, xor_yield=None, do_yield=True):
        if memo is None:
            memo = set()
        if do_yield:
            for name, p in parent_module._parameters.items():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p
        for mname, module in parent_module.named_children():
            if do_filter:
                if hasattr(module, 'sparse') and module.sparse:
                    do_yield = xor_yield
                else:
                    do_yield = not xor_yield
            else:
                do_yield = True
            submodule_prefix = prefix + ('.' if prefix else '') + mname
            for name, p in self.named_parameters(module, memo, submodule_prefix, do_filter, xor_yield, do_yield):
                yield name, p

    @property
    def batch_first(self):
        return getattr(self.model, 'batch_first', False)

    def iterate(self, data, training=True, data_loader=None):

        vola_after, \
        features, \
        presentation_toks_np, \
        question_1_toks_np, \
        answer_1_toks_np, = data

        batch_size = vola_after.size(0)
        items_in_batch = vola_after.size(0)

        if self.cuda and not isinstance(self.model_with_loss, DataParallel):
            vola_after = vola_after.to(self.devices[0])
            features = features.to(self.devices[0])
            presentation_toks_np = presentation_toks_np.to(self.devices[0])
            question_1_toks_np = question_1_toks_np.to(self.devices[0])
            answer_1_toks_np = answer_1_toks_np.to(self.devices[0])

        inputs = (
            features,
            presentation_toks_np,
            question_1_toks_np,
            answer_1_toks_np,
        )

        # print(inputs)
        # print(labels)

        # compute output

        if training:
            backward_loss, select_loss, output = self.model_with_loss(inputs, vola_after)
        else:
            with torch.no_grad():
                backward_loss, select_loss, output = self.model_with_loss(inputs, vola_after)

        backward_loss = backward_loss.sum()/(1 if not isinstance(self.devices, list) and not isinstance(self.devices, tuple) else len(self.devices))
        select_loss = select_loss.sum()/(1 if not isinstance(self.devices, list) and not isinstance(self.devices, tuple) else len(self.devices))

        if training:
            for optimizer in self.optimizers:
                # compute gradient and do SGD step
                optimizer.zero_grad()
            backward_loss.backward()
            for optimizer in self.optimizers:
                if self.grad_clip is not None and self.grad_clip > 0:
                    clip_grad_norm(self.model.parameters(), self.grad_clip)
                optimizer.step()

        return (
            select_loss.item(),
            items_in_batch,
            output.data.cpu().clamp(min=0, max=float('inf')).numpy(),
            vola_after.data.view(batch_size).cpu().numpy()
        )

    def _feed_data(self, data_loader, num_iterations=None, training=True):
        if training:
            counter = cycle(range(self.keep_checkpoints))
            assert self.optimizers is not None

        num_iterations = num_iterations or len(data_loader) - 1
        batch_time = AverageMeter()
        data_time = AverageMeter()
        tok_time = AverageMeter()
        losses = AverageMeter()
        spearmanrs = ScipyStatsMeter(scipy.stats.spearmanr)
        kendalltaus = ScipyStatsMeter(scipy.stats.kendalltau)
        pearsons = ScipyStatsMeter(scipy.stats.pearsonr)
        pearsons_log = ScipyStatsMeter(scipy.stats.pearsonr, pre_proc=numpy.log1p)
        r2_score = SkLearnMetricMeter(sklearn.metrics.r2_score)
        mean_absolute_error = SkLearnMetricMeter(sklearn.metrics.mean_absolute_error)
        root_mean_squared_error = SkLearnMetricMeter(sklearn.metrics.mean_squared_error, post_proc=math.sqrt)

        end = time.time()

        for i, data in enumerate(data_loader):

            # measure data loading time
            data_time.update(time.time() - end)

            if training:
                self._epoch += 1. / len(data_loader)
                self.training_steps += 1
                # update optimizer according to epoch and steps
                for optimizer in self.optimizers:
                    optimizer.update(self.epoch, self.training_steps)

            # do a train/evaluate iteration
            loss, items_in_batch, predictions, labels = self.iterate(data, training=training, data_loader=data_loader)

            # measure accuracy and record loss
            losses.update(loss, items_in_batch)
            spearmanrs.update(predictions, labels)
            kendalltaus.update(predictions, labels)
            pearsons.update(predictions, labels)
            pearsons_log.update(predictions, labels)
            r2_score.update(predictions, labels)
            mean_absolute_error.update(predictions, labels)
            root_mean_squared_error.update(predictions, labels)

            # measure elapsed time
            elapsed = time.time() - end
            batch_time.update(elapsed)
            tok_time.update(items_in_batch / elapsed, items_in_batch)

            end = time.time()
            last_iteration = (i == (len(data_loader) - 1))
            if i > 0 or last_iteration:
                if i % self.print_freq == 0 and training or last_iteration:
                    best_loss = self.valid_loss if not training else 0
                    logging.info('{phase} - Epoch: [{0}][{1}/{2}]\t'
                                 'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                                 # 'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                                 'SR {spearmanrs.avg:.3f}\t'
                                 'KT {kendalltaus.avg:.3f}\t'
                                 'PR {pearsons.avg:.3f}\t'
                                 'R2 {r2_score.avg:.3f}\t'
                                 'MAE {mean_absolute_error.avg:.3f}\t'
                                 'RMSE {root_mean_squared_error.avg:.3f}\t'
                                 'Items/sec {tok_time.val:.3f} ({tok_time.avg:.3f})\t'
                                 'Loss {loss.val:.4f} ({loss.avg:.4f} / {best_loss:.4f})\t'.format(
                                    int(self.epoch), i, len(data_loader),
                                    phase='TRAINING' if training else 'EVALUATING',
                                    batch_time=batch_time, data_time=data_time, tok_time=tok_time,
                                    loss=losses, best_loss=best_loss,
                                    spearmanrs=spearmanrs,
                                    kendalltaus=kendalltaus,
                                    pearsons=pearsons,
                                    r2_score=r2_score,
                                    mean_absolute_error=mean_absolute_error,
                                    root_mean_squared_error=root_mean_squared_error, ))
                if training and (i % self.save_freq == 0 or last_iteration and self.save_epoch_freq == 0):
                    self.save(identifier=next(counter))
                if num_iterations > 0 and i % num_iterations == 0 or last_iteration:
                    yield {
                        'loss': losses.avg,
                        'spearmanr': spearmanrs.avg,
                        'kendalltau': kendalltaus.avg,
                        'pearsons': pearsons.avg,
                        'pearsons_log': pearsons_log.avg,
                        'mean_absolute_error': mean_absolute_error.avg,
                        'r2_score': r2_score.avg,
                        'root_mean_squared_error': root_mean_squared_error.avg,
                    }
                    losses.reset()

    def optimize(self, train_data_loader):
        # switch to train mode
        self.model.train()
        for result in self._feed_data(
                train_data_loader,
                num_iterations=self.eval_freq,
                training=True):
            yield result
            self.model.train()

    def evaluate(self, eval_data_loader):
        # switch to evaluate mode
        self.model.eval()
        with torch.no_grad():
            for r in self._feed_data(
                    eval_data_loader,
                    training=False):
                result = r
            return result

    def run(self, train_loader, val_loader=None):
        self.save_epoch_freq = next(self.save_epoch_freq_cycle)
        for train_results in self.optimize(train_loader):
            results = {'epoch': self.epoch,
                       'training_steps': self.training_steps,
                       'training_loss': train_results['loss'],
                       }
            if val_loader is not None:
                # evaluate on validation set
                val_results = self.evaluate(val_loader)

                is_best = False
                best_select_metric = list()

                self.average_valid_loss.update(val_results['loss'])

                if val_results['spearmanr'] > self.valid_sr:
                    self.valid_sr = val_results['spearmanr']
                    if 'max_sr' in self.model_select_metric:
                        best_select_metric.append('max_sr')
                        is_best = True

                if val_results['kendalltau'] > self.valid_kt:
                    self.valid_kt = val_results['kendalltau']
                    if 'max_kt' in self.model_select_metric:
                        best_select_metric.append('max_kt')
                        is_best = True

                if val_results['pearsons'] > self.valid_pr:
                    self.valid_pr = val_results['pearsons']
                    if 'max_pr' in self.model_select_metric:
                        best_select_metric.append('max_pr')
                        is_best = True

                if val_results['root_mean_squared_error'] < self.valid_rmse:
                    self.valid_rmse = val_results['root_mean_squared_error']
                    if 'max_rmse' in self.model_select_metric:
                        best_select_metric.append('max_rmse')
                        is_best = True

                if val_results['loss'] < self.valid_loss:
                    best_select_metric.append('min_loss')
                    self.valid_loss = val_results['loss']
                    is_best = True

                if is_best:
                    if self.save_epoch_freq == 0:
                        self.save(save_all=True, is_best=True, tags=best_select_metric, identifier=next(self.counter))

                results['validation_loss'] = val_results['loss']
                results['validation_spearmanr'] = val_results['spearmanr']
                results['validation_kendalltau'] = val_results['kendalltau']
                results['validation_pearsons'] = val_results['pearsons']
                results['validation_pearsons_log'] = val_results['pearsons_log']
                results['r2_score'] = val_results['r2_score']
                results['mean_absolute_error'] = val_results['mean_absolute_error']
                results['root_mean_squared_error'] = val_results['root_mean_squared_error']

                if self.last_val_loss is None:
                    self.last_val_loss = val_results['loss']
                else:
                    self.moving_average_criterion_change = running_mean(math.fabs((self.last_val_loss - val_results['loss'])/val_results['loss']), self.moving_average_criterion_change)

                if val_results['loss'] > self.patience_criterion_treshold or \
                    (self.moving_average_criterion_change is not None and self.moving_average_criterion_change < self.patience_criterion_change) or \
                    val_results['loss'] > self.valid_loss:
                    # print("val_results['loss'] > self.patience_criterion_treshold", val_results['loss'] > self.patience_criterion_treshold)
                    # print("moving_average_criterion_change", (self.moving_average_criterion_change is not None and self.moving_average_criterion_change < self.patience_criterion_change))
                    # print("val_results['loss'] > self.valid_loss", val_results['loss'] > self.valid_loss)
                    if self.epoch == self.terminate_epochs:
                        self.terminate = True
                    elif self.epoch > self.terminate_epochs:
                        # set new timer
                        self.terminate_epochs = self.epoch + self.patience_epochs
                else:
                    self.terminate_epochs = self.epoch + self.patience_epochs

            self.results.add(**results)
            if self.save_epoch_freq == 0:
                self.results.save()

            self.save_epoch_freq = next(self.save_epoch_freq_cycle)

    def init_optimizer(self):
        self.optimizers = list()
        if type(self.regime) is dict:
            for params_regex, regime in self.regime.items():
                print(list(map(lambda name_param: name_param[0], filter(lambda name_param: name_param[1].requires_grad and re.match(params_regex, name_param[0]), self.model.named_parameters()))))
                self.optimizers.append(OptimRegime(map(lambda name_param: name_param[1], filter(lambda name_param: name_param[1].requires_grad and re.match(params_regex, name_param[0]), self.model.named_parameters())), regime=regime))
        elif type(self.regime) is list:
            print("Trainable parameters: ", list(map(lambda name_param: name_param[0], filter(lambda name_param: name_param[1].requires_grad, self.model.named_parameters()))))
            self.optimizers.append(OptimRegime(filter(lambda p: p.requires_grad, self.model.parameters()), regime=self.regime))

    def load_state_dict(self, model: torch.nn.Module, state_dict, strict=True, freeze_param=False, weight_map=None, main_gpu=0):
        """Copies parameters and buffers from :attr:`state_dict` into
        this module and its descendants. If :attr:`strict` is ``True`` then
        the keys of :attr:`state_dict` must exactly match the keys returned
        by this module's :func:`state_dict()` function.

        Arguments:
            state_dict (dict): A dict containing parameters and
                persistent buffers.
            strict (bool): Strictly enforce that the keys in :attr:`state_dict`
                match the keys returned by this module's `:func:`state_dict()`
                function.
        """

        # own_state = dict(model.state_dict().items())
        own_state = model.state_dict(keep_vars=True)
        loaded_params = list()

        for name, param in state_dict.items():
            if weight_map is not None and name in weight_map:
                print('Mapping from {} to {}'.format(name, weight_map[name]))
                name = weight_map[name]
            if name in own_state:
                if isinstance(param, torch.nn.Parameter):
                    # backwards compatibility for serialized parameters
                    param_data = param.data
                else:
                    param_data = param
                try:
                    if isinstance(own_state[name], torch.nn.Parameter):
                        own_state[name].data.copy_(param_data)
                    else:
                        own_state[name].copy_(param_data)
                    if type(freeze_param) is list and (name in freeze_param or type(freeze_param[0]) is str and freeze_param[0]=='True'):
                        logging.info("Freezing {}".format(name))
                        own_state[name].requires_grad = False
                except Exception as e:
                    print(e)
                    raise RuntimeError('While copying the parameter named {}, '
                                       'whose dimensions in the model are {} and '
                                       'whose dimensions in the checkpoint are {}.'
                                       .format(name, own_state[name].size(), param.size()))
                loaded_params.append(name)
            elif strict:
                raise KeyError('unexpected key "{}" in state_dict'
                               .format(name))
            else:
                print('{} missing from state_dict'.format(name))
                pass
        missing = set(own_state.keys()) - set(state_dict.keys())
        if strict:
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))
        print('loaded from state_dict: "{}"'.format(loaded_params))
        # print('missing keys in state_dict: "{}"'.format(missing))

        for mod in model.modules():
            if hasattr(mod, 'flatten_parameters'):
                mod.flatten_parameters()

    def load(self, filename, as_pretraining=True, resume_filter=None, freeze_param=False, weight_map=None):
        if os.path.isfile(filename):
            if isinstance(self.devices, tuple) or isinstance(self.devices, list):
                main_device = self.devices[0]
            else:
                main_device = self.devices
            checkpoint = torch.load(filename, map_location={
                'cuda:0': "cuda:{}".format(main_device),
                'cuda:1': "cuda:{}".format(main_device),
                'cuda:2': "cuda:{}".format(main_device),
                'cuda:3': "cuda:{}".format(main_device),
            })
            state_dict = checkpoint['state_dict']
            if resume_filter is not None:
                state_dict = {k: v for k, v in state_dict.items() if k in resume_filter}
            self.load_state_dict(self.model, state_dict, strict=False, freeze_param=freeze_param, weight_map=weight_map)
            if not as_pretraining:
                self.init_optimizer()
                self._epoch = checkpoint['epoch']
                self.training_steps = checkpoint['training_steps']
                self.validation_results = checkpoint['validation_results']
                for optim, optim_state_dict in zip(self.optimizers, checkpoint['optimizer_state_dict']):
                    optim.load_state_dict(optim_state_dict)
            else:
                self.init_optimizer()
            logging.info("loaded checkpoint '%s' (epoch %s)", filename, self.epoch)

        else:
            logging.error('invalid checkpoint: {}'.format(filename))

    def save(self, filename=None, identifier=None, is_best=False, save_all=False, tags=None, keep_last=5):
        state = {
            'epoch': self._epoch,
            'training_steps': self.training_steps,
            'state_dict': self.model.state_dict(),
            'optimizer_state_dict': [optim.state_dict() for optim in self.optimizers],
            'validation_results': getattr(self, 'validation_results', None),
        }
        state = dict(list(state.items()) + list(self.save_info.items()))
        identifier = identifier or ''
        filename = filename or self.checkpoint_filename % identifier
        filename = os.path.join(self.save_path, filename)
        logging.info('saving model to %s' % filename)
        torch.save(state, filename)
        if is_best:
            if tags is not None:
                for tag in tags:
                    mb_fn = 'model_best-{}.pth.tar'.format(tag)
                    if os.path.exists(os.path.join(self.save_path, mb_fn)):
                        mb_fn_epoch = 'model_best-{}-{}.pth.tar'.format(tag, identifier)
                        shutil.copyfile(os.path.join(self.save_path, mb_fn), os.path.join(self.save_path, mb_fn_epoch))
                    shutil.copyfile(filename, os.path.join(self.save_path, mb_fn))
            else:
                mb_fn = 'model_best.pth.tar'
                if os.path.exists(os.path.join(self.save_path, mb_fn)):
                    mb_fn_epoch = 'model_best-{}.pth.tar'.format(identifier)
                    shutil.copyfile(os.path.join(self.save_path, mb_fn), os.path.join(self.save_path, mb_fn_epoch))
                shutil.copyfile(filename, os.path.join(self.save_path, 'model_best.pth.tar'))
        if save_all:
            shutil.copyfile(filename, os.path.join(
                self.save_path, 'checkpoint_epoch_%s.pth.tar' % self.epoch))


