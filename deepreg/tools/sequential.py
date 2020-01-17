import torch
from collections import OrderedDict


class Sequential(torch.nn.Sequential):

    def __init__(self, *args):
        super(Sequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            discount_none = 0
            for idx, module in enumerate(args):
                if module is not None:
                    self.add_module(str(idx-discount_none), module)
                else:
                    discount_none += 1

    def forward(self, input):
        for module in self._modules.values():
            if type(input) == dict:
                input = module(input)
            else:
                input = module(input)
        return input