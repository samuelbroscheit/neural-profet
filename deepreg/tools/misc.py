import random
import numpy as np

def set_global_seeds(i):
    try:
        import torch
    except ImportError:
        pass
    else:
        torch.manual_seed(i)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(i)
    np.random.seed(i)
    random.seed(i)

def prettyformat_dict_string(d, indent=''):
    result = list()
    for k, v in d.items():
        if isinstance(v, dict):
            result.append('{}{}:\t\n{}'.format(indent, k, prettyformat_dict_string(v, indent + '  ')))
        else:
            result.append('{}{}:\t{}\n'.format(indent, k, v))
    return ''.join(result)

