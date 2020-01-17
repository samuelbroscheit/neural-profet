#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Grid-based Hyperparamter Optimization:

Takes as input two dictionaries, model parameters
and training parameters, from which a grid is created. We train each
configuration of the grid for 'search_epochs' (may 'repeat_sample' times) and
use the resulting performance metric to pick the best seen configuration which
is then trained for 'final_epochs'.
"""

import ast
from datetime import datetime
import os

import numpy
import pandas
from ast import literal_eval
from collections import OrderedDict
from copy import deepcopy

from sklearn.model_selection import ParameterGrid

from deepreg.train import parser, main

parser.add_argument('--skip',  default=0, type=int, help='Skip this many experiments')
parser.add_argument('--stop',  default=float('inf'), type=int, help='Stop at this many experiments')
parser.add_argument('--search_epochs',  default=20, type=int, help='Number of model parameter samples')
parser.add_argument('--final_epochs',  default=100, type=int, help='Number of model parameter samples')
parser.add_argument('--repeat_sample',  default=1, type=int, help='Number of repetition for each sample')


def isnumeric(n):
    try:
        float(n)
        if isinstance(n, bool):
            return False
        return True
    except:
        return False

def make_type_dict(a_dict):
    result = dict()
    for k,v in a_dict.items():
        if type(v[0]) != bool:
            result[k]= type(v[0])
        else:
            result[k] = ast.literal_eval
    return result

def convert_row_to_dict(row, type_dict):
    result = dict()
    for k,v in row.to_dict().items():
        parts = k.split('#')
        if len(parts) == 1:
            if k not in type_dict:
                continue
            result[k] = type_dict[k](v)
        else:
            k_,v_ = parts
            if k_ not in type_dict:
                continue
            if v == 1.0:
                result[k_] = type_dict[k_](v_)
    return result


def run_setting(orig_args, hyper_experiment_nr, results, train_cell, model_cell, train_param_cols, model_param_cols, eval_args):

    hyper_experiment_nr += 1
    if hyper_experiment_nr <= orig_args.skip:
        return

    args = deepcopy(orig_args)

    args.optimization_config = [{'epoch': 0, 'optimizer': train_cell['optimizer'], 'lr': train_cell['lr'], 'weight_decay': train_cell['weight_decay']}]

    for k, v in train_cell.items():
        if k in args.__dict__:
            args.__dict__[k] = v

    for k, v in model_cell.items():
        if k in args.model_config:
            args.model_config[k] = v

    best_valid_loss, average_valid_loss, valid_sr, valid_kt, valid_pr, valid_rmse = main(args, 'hyper#{}'.format(hyper_experiment_nr))

    select_score = valid_pr

    results.append([train_cell[k] for k in train_param_cols] + [model_cell[k] for k in model_param_cols] + [select_score, best_valid_loss, valid_sr, valid_kt, valid_pr, valid_rmse])
    return hyper_experiment_nr, results

select_score_ascending=False

def results_to_dataframe(results, train_param_cols, model_param_cols, categorial_cols, val_loss_col, eval_cols):
    df = pandas.DataFrame(results, columns=train_param_cols + model_param_cols + val_loss_col + eval_cols)
    df = pandas.get_dummies(df, columns=categorial_cols, prefix_sep='#', )
    columns_list_without_loss = list(df.columns.tolist())
    for c in val_loss_col + eval_cols:
        columns_list_without_loss.remove(c)
    columns_list_with_loss = columns_list_without_loss + val_loss_col + eval_cols
    df = df.sort_values(by=val_loss_col, ascending=select_score_ascending)
    return df, columns_list_without_loss, columns_list_with_loss


def hyper_grid(
        model_param_grid,
        train_param_grid,
        orig_args,
        eval_args=None,
        eval_cols=None,
):

    time_stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    config_name = os.path.splitext(os.path.basename(orig_args.config))[0]

    model_param_type = make_type_dict(model_param_grid)
    train_param_type = make_type_dict(train_param_grid)

    orig_args.results_dir = os.path.join(orig_args.results_dir, 'hyper', config_name, time_stamp)

    joined_type_dict = dict(list(model_param_type.items()) + list(train_param_type.items()))

    model_param_grid = OrderedDict({k:v for k,v in model_param_grid.items() if k in orig_args.model_config})
    train_param_grid = OrderedDict(train_param_grid)

    print(model_param_grid)

    results = list()
    header = None

    val_loss_col = ['select_score']

    model_param_cols = ['{}'.format(k) for k, v in model_param_grid.items()]
    train_param_cols = ['{}'.format(k) for k, v in train_param_grid.items()]

    numeric_cols = [k for k in train_param_cols if isnumeric(train_param_grid[k][0])] + \
                   [k for k in model_param_cols if isnumeric(model_param_grid[k][0])]

    categorial_cols = [k for k in train_param_cols if not isnumeric(train_param_grid[k][0])] + \
                   [k for k in model_param_cols if not isnumeric(model_param_grid[k][0])]

    all_settings = list()
    for train_cell_predict in ParameterGrid(train_param_grid):
        for model_cell_predict in ParameterGrid(model_param_grid):
            all_settings.append([train_cell_predict[k] for k in train_param_cols] + [model_cell_predict[k] for k in model_param_cols])
    predict_df = pandas.DataFrame(all_settings, columns=train_param_cols + model_param_cols)
    predict_df = pandas.get_dummies(predict_df, columns=categorial_cols, prefix_sep='#', )

    print(len(predict_df))

    hyper_experiment_nr = 0
    columns_list_without_loss = []

    for train_cell in ParameterGrid(train_param_grid):
        for model_cell in ParameterGrid(model_param_grid):
            for repeat in range(orig_args.repeat_sample):

                train_cell['epochs'] = orig_args.search_epochs
                hyper_experiment_nr, results = run_setting(orig_args, hyper_experiment_nr, results, train_cell, model_cell, train_param_cols, model_param_cols, eval_args)
                df, columns_list_without_loss, columns_list_with_loss = results_to_dataframe(results, train_param_cols, model_param_cols, categorial_cols, val_loss_col, eval_cols)
                df[columns_list_with_loss].to_csv(os.path.join(orig_args.results_dir, '{}-{}.{}'.format(config_name, time_stamp, 'csv')))

    tmp_df = df[~df.isin([numpy.nan, numpy.inf, -numpy.inf]).any(1)]

    print(tmp_df.iloc[0, :])

    final_dict = convert_row_to_dict(tmp_df.iloc[0, :], joined_type_dict)
    final_dict['epochs'] = orig_args.final_epochs

    hyper_experiment_nr, results = run_setting(orig_args, hyper_experiment_nr, results, final_dict, final_dict, train_param_cols, model_param_cols, eval_args)
    df, columns_list_without_loss, columns_list_with_loss = results_to_dataframe(results, train_param_cols, model_param_cols, categorial_cols, val_loss_col, eval_cols)
    df[columns_list_with_loss].to_csv(os.path.join(orig_args.results_dir, '{}-{}.{}'.format(config_name, time_stamp, 'csv')))
    df[columns_list_with_loss].to_csv(os.path.join(orig_args.results_dir, '{}-{}.{}'.format(config_name, time_stamp, 'csv')))


if __name__ == '__main__':

    args_ = parser.parse_args()
    if isinstance(args_.model_config, str):
        args_.model_config = literal_eval(args_.model_config)
    if isinstance(args_.train_data_config, str):
        args_.train_data_config = literal_eval(args_.train_data_config)
    if isinstance(args_.devices, str):
        args_.devices = literal_eval(args_.devices)
    orig_args = deepcopy(args_)

    model_param_grid = {
        'pretrained_embeddings_name': [
                                       'fasttext_200.wv.vectors.npy',
                                       ],
        'dropout': [0.0, ],
        'activation': ['ReLU', ],
        'hidden_size': [128, ],
        'hidden_layers': [1, ],
        'ctxt_window': [1, 2, 3,],
        'lstm_hidden_size': [100,],
        'lstm_num_layers': [1,],
        'adjust_embeddings': [False],
        'in_bn': [True,],
        'hid_bn': [False, ],
        'out_bn': [True, ],
        'with_finance_feat': [True],
        'sparse': [False],
    }

    train_param_grid = {
        'optimizer': ['Adagrad'],
        'lr': [0.05, ],
        'weight_decay': [0.0001, ],
        'batch_size': [112 * 1 if isinstance(orig_args.devices, int) else 112 * len(orig_args.devices)],
        'loss': ['ModelWrappedWithMSELoss',]
        # 'loss': ['ModelWrappedWithMSELossAndMulticlassBatchLoss']
    }

    hyper_grid(
            model_param_grid,
            train_param_grid,
            orig_args,
            eval_cols=['val_loss', 'SR', 'KT', 'PR', 'RMSE', ],
    )
