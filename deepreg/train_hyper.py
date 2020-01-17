#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Grid-based Bayesian Hyperparameter-Optimization:

Takes as input two dictionaries, model parameters and training parameters, from
which a grid of all possible hyperparameter-permutations is created. The reason
of splitting the model and training parameters is that we could (if we wanted to)
fit different model configurations for one and the same training configuration.

To initialize the Bayesian Optimization we first sample 'random_exploration_iter'
many random configurations from the grid and train each configuration for
'search_epochs' times and use the resulting performance metric to initialize a
GP regressor. We fit the GP on the current known configurations and then score
the whole grid and select the unseen grid configuration the has the best (either
largest/smallest) predicted performance metric. The latter procedure is then repeated
for 'train_iter*model_iter*repeat_sample' times. The best seen configuration is
then trained for 'final_epochs'.
"""

import ast
from datetime import datetime
import os

import numpy
import pandas
from ast import literal_eval
from collections import OrderedDict
from copy import deepcopy

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import ParameterGrid, ParameterSampler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from deepreg.train import parser, main

parser.add_argument('--skip',  default=0, type=int, help='Skip this many experiments')
parser.add_argument('--stop',  default=float('inf'), type=int, help='Stop at this many experiments')
parser.add_argument('--train_iter',  default=35, type=int, help='Number of training parameter samples')
parser.add_argument('--model_iter',  default=2, type=int, help='Number of model parameter samples')
parser.add_argument('--random_exploration_iter',  default=10, type=int, help='Number of random exploration steps before actively exploring')
parser.add_argument('--refine_from_top_k',  default=20, type=int, help='Refine model in the top k')
parser.add_argument('--search_epochs',  default=20, type=int, help='Number of model parameter samples')
parser.add_argument('--final_epochs',  default=100, type=int, help='Number of model parameter samples')
parser.add_argument('--repeat_sample',  default=1, type=int, help='Number of repetition for each sample')
parser.add_argument('--resume_from_csv',  type=str, help='Load previous search from csv')


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

    args = deepcopy(orig_args)

    args.optimization_config = [{'epoch': 0, 'optimizer': train_cell['optimizer'], 'lr': train_cell['lr'], 'weight_decay': train_cell['weight_decay']}]

    for k, v in train_cell.items():
        if k in args.__dict__:
            args.__dict__[k] = v

    for k, v in model_cell.items():
        if k in args.model_config:
            args.model_config[k] = v

    best_valid_loss, average_valid_loss, valid_sr, valid_kt, valid_pr, valid_rmse = main(args, 'hyper#{}'.format(hyper_experiment_nr))

    select_score = best_valid_loss

    results.append([train_cell[k] for k in train_param_cols] + [model_cell[k] for k in model_param_cols] + [select_score, best_valid_loss, valid_sr, valid_kt, valid_pr, valid_rmse])
    return hyper_experiment_nr, results

select_score_ascending=True

def results_to_dataframe(results, train_param_cols, model_param_cols, categorial_cols, val_loss_col, eval_cols):
    df = pandas.DataFrame(results, columns=train_param_cols + model_param_cols + val_loss_col + eval_cols)
    df = pandas.get_dummies(df, columns=categorial_cols, prefix_sep='#', )
    columns_list_without_loss = list(df.columns.tolist())
    for c in val_loss_col + eval_cols:
        columns_list_without_loss.remove(c)
    columns_list_with_loss = columns_list_without_loss + val_loss_col + eval_cols
    df = df.sort_values(by=val_loss_col, ascending=select_score_ascending)
    return df, columns_list_without_loss, columns_list_with_loss

def dataframe_to_results(df:pandas.DataFrame):
    results = list()
    for item in df.itertuples():
        results.append([item[i] for i in range(1,len(item))])
    return results


def hyper_opt(
        model_param_grid,
        train_param_grid,
        orig_args,
        eval_args=None,
        eval_cols=None,
        df=None
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

    if df is not None:
        results = dataframe_to_results(df)
    else:
        results = list()

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
    columns_list_without_loss = list(predict_df.columns.tolist())

    if df is not None:
        orig_args.skip = len(df)

    for train_cell in ParameterSampler(train_param_grid, n_iter=orig_args.train_iter):
        for model_cell in ParameterSampler(model_param_grid, n_iter=orig_args.model_iter):
            for repeat in range(orig_args.repeat_sample):

                hyper_experiment_nr += 1
                if hyper_experiment_nr <= orig_args.skip or hyper_experiment_nr > orig_args.stop:
                    continue

                if hyper_experiment_nr >= orig_args.random_exploration_iter:

                    tmp_df = df[~df.isin([numpy.nan, numpy.inf, -numpy.inf]).any(1)]

                    features = tmp_df[columns_list_without_loss].astype(float)
                    response = tmp_df[val_loss_col].values.astype(float).reshape(-1)

                    linreg = Pipeline([
                        ('s', StandardScaler()),
                        ('r', GaussianProcessRegressor(kernel=1.0 * RBF(1.0), alpha=1e-10, normalize_y=True, n_restarts_optimizer=20)),
                    ])

                    print("linreg.fit")
                    linreg.fit(features, response)

                    predict_df[val_loss_col[0]], predict_df['std'] = linreg.predict(predict_df[columns_list_without_loss].astype(float), return_std=True)
                    if select_score_ascending:
                        predict_df['select_score+std'] = predict_df['select_score'] - predict_df['std']
                    else:
                        predict_df['select_score+std'] = predict_df['select_score'] + predict_df['std']
                    print('Current best setting:')
                    print(df.iloc[0, :])

                    # explore_df = predict_df[:orig_args.refine_from_top_k].sort_values(by='std', ascending=False)

                    predict_df = predict_df.sort_values(by='select_score+std', ascending=select_score_ascending)
                    explore_top_best_config = 0
                    while (predict_df[columns_list_without_loss].iloc[explore_top_best_config, :] == df[columns_list_without_loss]).all(1).any():
                        explore_top_best_config += 1

                    print("picking {}".format(explore_top_best_config))

                    explore_dict = convert_row_to_dict(predict_df[columns_list_without_loss].iloc[explore_top_best_config, :], joined_type_dict)
                    explore_dict['epochs'] = orig_args.search_epochs

                    print('\nExplore:')
                    print(predict_df[columns_list_without_loss].iloc[explore_top_best_config, :])

                    hyper_experiment_nr, results = run_setting(orig_args, hyper_experiment_nr, results, explore_dict, explore_dict, train_param_cols, model_param_cols, eval_args)
                    df, columns_list_without_loss, columns_list_with_loss = results_to_dataframe(results, train_param_cols, model_param_cols, categorial_cols, val_loss_col, eval_cols)
                    df[columns_list_with_loss].to_csv(os.path.join(orig_args.results_dir, '{}-{}.{}'.format(config_name, time_stamp, 'csv')))

                    predict_df.to_csv(os.path.join(orig_args.results_dir, '{}-{}-{}.{}'.format(config_name, time_stamp, 'predicted', 'csv')))

                else:

                    train_cell['epochs'] = orig_args.search_epochs
                    hyper_experiment_nr, results = run_setting(orig_args, hyper_experiment_nr, results, train_cell, model_cell, train_param_cols, model_param_cols, eval_args)
                    df, columns_list_without_loss, columns_list_with_loss = results_to_dataframe(results, train_param_cols, model_param_cols, categorial_cols, val_loss_col, eval_cols)
                    df[columns_list_with_loss].to_csv(os.path.join(orig_args.results_dir, '{}-{}.{}'.format(config_name, time_stamp, 'csv')))

    tmp_df = df[~df.isin([numpy.nan, numpy.inf, -numpy.inf]).any(1)]

    features = tmp_df[columns_list_without_loss].astype(float)
    response = tmp_df[val_loss_col].values.astype(float).reshape(-1)

    linreg = Pipeline([
        ('s', StandardScaler()),
        ('r', GaussianProcessRegressor(kernel=1.0 * RBF(1.0), alpha=1e-10, normalize_y=True, n_restarts_optimizer=20)),
    ])

    print("linreg.fit")
    linreg.fit(features, response)

    print('Final best setting:')
    predict_df[val_loss_col[0]], predict_df['std'] = linreg.predict(predict_df[columns_list_without_loss].astype(float), return_std=True)
    predict_df['select_score*std'] = predict_df['select_score'] * predict_df['std']
    predict_df = predict_df.sort_values(by='select_score', ascending=select_score_ascending)
    predict_df = predict_df[0:20].sort_values(by='select_score*std', ascending=select_score_ascending)
    print(predict_df.iloc[0, :])

    final_dict = convert_row_to_dict(predict_df.iloc[0, :], joined_type_dict)
    final_dict['epochs'] = orig_args.final_epochs

    hyper_experiment_nr, results = run_setting(orig_args, hyper_experiment_nr, results, final_dict, final_dict, train_param_cols, model_param_cols, eval_args)
    df, columns_list_without_loss, columns_list_with_loss = results_to_dataframe(results, train_param_cols, model_param_cols, categorial_cols, val_loss_col, eval_cols)
    df[columns_list_with_loss].to_csv(os.path.join(orig_args.results_dir, '{}-{}.{}'.format(config_name, time_stamp, 'csv')))
    df[columns_list_with_loss].to_csv(os.path.join(orig_args.results_dir, '{}-{}.{}'.format(config_name, time_stamp, 'csv')))


if __name__ == '__main__':

    args_ = parser.parse_args()
    if isinstance(args_.model_config, str):
        args_.model_config = literal_eval(args_.model_config)
    if isinstance(args_.devices, str):
        args_.devices = literal_eval(args_.devices)
    if isinstance(args_.train_data_config, str):
        args_.train_data_config = literal_eval(args_.train_data_config)
    orig_args = deepcopy(args_)

    if orig_args.resume_from_csv:
        df = pandas.read_csv(orig_args.resume_from_csv, index_col=0)
    else:
        df = None

    model_param_grid = {
        'pretrained_embeddings_name': ['fasttext_100.wv.vectors.npy',
                                       'fasttext_200.wv.vectors.npy',
                                       ],
        'dropout': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, ],
        'activation': ['ReLU', 'Tanh',],
        # 'hidden_size': [128, 256, 512, 1024, 2048],
        'hidden_size': [128, 256],
        'hidden_layers': [0, 1, 2, 3,],
        # 'ctxt_window': [1, 2, 3,],
        'lstm_hidden_size': [50, 100,],
        'lstm_num_layers': [1, 2, ],
        'adjust_embeddings': [True],
        'in_bn': [True, False],
        'hid_bn': [True, False],
        'out_bn': [True, False],
        'sparse': [False, True],
    }

    train_param_grid = {
        'optimizer': ['Adagrad'],
        'lr': [5e-2, 1e-2, 5e-3, ],
        'weight_decay': [1e-4, 1e-5, 1e-6, 1e-7, ],
        'batch_size': [256],
        'loss': ['ModelWrappedWithMSELoss',]
    }

    hyper_opt(
            model_param_grid,
            train_param_grid,
            orig_args,
            eval_cols=['val_loss', 'SR', 'KT', 'PR', 'RMSE', ],
            df=df,
    )
