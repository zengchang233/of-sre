#  from torch import nn
#  import torch.nn.functional as F
#  import torch
import oneflow as of
import oneflow.nn as nn
import oneflow.nn.functional as F
import math

class ReLU20(nn.Hardtanh): # relu
    def __init__(self, inplace=False):
        super(ReLU20, self).__init__(0, 20, inplace)

    def extra_repr(self):
        inplace_str = 'inplace' if self.inplace else ''
        return inplace_str
