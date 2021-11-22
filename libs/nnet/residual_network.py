import sys
import math
sys.path.insert(0, '../../')

import oneflow as of
import oneflow.nn as nn
import oneflow.nn.functional as F
#  from torch import nn
#  import torch.nn.functional as F
#  import torch

from libs.components.conv import conv3x3
from libs.components.activation import ReLU20
from libs.components import pooling

class BasicBlock(nn.Module): # 定义block
    expansion = 1
    def __init__(self, in_channels, channels, stride=1, downsample=None): # 输入通道，输出通道，stride，下采样
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, channels, stride)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = ReLU20(inplace=True)
        self.conv2 = conv3x3(channels, channels)
        self.bn2 = nn.BatchNorm2d(channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out # block输出

class SpeakerEmbNet(nn.Module): # 定义resnet
    def __init__(self, opts): # block类型，embedding大小，分类数，maigin大小
        super(SpeakerEmbNet, self).__init__()
        input_channel = 1
        input_dim = opts['input_dim']
        hidden_dim = opts['hidden_dim']
        residual_block_layers = opts['residual_block_layers']
        fc_layers = opts['fc_layers']
        block = BasicBlock
        embedding_dim = opts['embedding_dim']
        num_head = opts['num_head']
        #  pooling = opts['pooling']
        self.relu = ReLU20(inplace=True)

        block_layers = []
        for dim, block_layer in zip(hidden_dim, residual_block_layers):
            block_layers.append(nn.Conv2d(input_channel, dim, kernel_size = 5, stride = 2, padding = 2, bias = False))
            block_layers.append(nn.BatchNorm2d(dim))
            #  block_layers.append(ReLU20(inplace = True))
            block_layers.append(self._make_layer(block, dim, block_layer))
            input_channel = dim

        self.residual = nn.Sequential(*block_layers)

        residual_output_shape = math.ceil(input_dim / (2 ** len(residual_block_layers)))
        if opts['pooling'] == 'STAT':
            self.pooling = pooling.STAT()
            self.fc1 = nn.Linear(hidden_dim[-1] * residual_output_shape * 2, embedding_dim)
        elif opts['pooling'] == 'ASP':
            self.pooling = pooling.AttentiveStatPooling(attention_hidden_size, hidden_dim[-1])
            self.fc1 = nn.Linear(hidden_dim[-1] * residual_output_shape * 2, embedding_dim)
            #  self.pooling = pooling.AttentiveStatisticsPooling(hidden_dim[-1], hidden_size = attention_hidden_size)
        elif opts['pooling'] == 'multi_head_ffa':
            self.pooling = pooling.MultiHeadFFA(hidden_dim[-1], attention_hidden_size)
            self.fc1 = nn.Linear(hidden_dim[-1] * residual_output_shape, embedding_dim)
        elif opts['pooling'] == 'multi_head_attention':
            self.pooling = pooling.MultiHeadAttentionPooling(residual_output_shape * hidden_dim[-1], num_head = num_head)
            self.fc1 = nn.Linear(hidden_dim[-1] * residual_output_shape * 2, embedding_dim)
        elif opts['pooling'] == 'multi_resolution_attention':
            self.pooling = pooling.MultiResolutionMultiHeadAttentionPooling(residual_output_shape * hidden_dim[-1], num_head = num_head)
            self.fc1 = nn.Linear(hidden_dim[-1] * residual_output_shape * 2, embedding_dim)
        else:
            raise NotImplementedError("Other pooling methods has been not implemented!")

        #  for m in self.modules(): # 对于各层参数的初始化
            #  if isinstance(m, nn.Conv2d): # 以2/n的开方为标准差，做均值为0的正态分布
            #      n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            #      m.weight.data.normal_(0, math.sqrt(2. / n))
            #  elif isinstance(m, nn.BatchNorm2d): # weight设置为1，bias为0
            #      m.weight.data.fill_(1)
            #      m.bias.data.zero_()
            #  elif isinstance(m, nn.BatchNorm1d): # weight设置为1，bias为0
            #      m.weight.data.fill_(1)
        #          m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        layers = [block(planes, planes, stride)]
        in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(in_planes, planes))
        return nn.Sequential(*layers)

    def extract_embedding(self, x):
        assert len(x.size()) == 3, "the shape of input must be 3 dimensions"
        x = x.unsqueeze(1)
        x = self.residual(x)
        x = x.reshape(x.shape[0], x.shape[1]*x.shape[2], x.shape[3])
        pooling_out = self.pooling(x)
        x = pooling_out.view(pooling_out.size(0), -1)
        x = self.fc1(x)
        return x, pooling_out

    def forward(self, x):
        '''
        params:
            x: input feature, B, C, T
        return:
            output of unnormalized speaker embedding
        '''
        x, _ = self.extract_embedding(x)
        return x

if __name__ == '__main__':
    import yaml
    from yaml import CLoader
    f = open('../../conf/model/resnet.yaml', 'r')
    opts = yaml.load(f, Loader = CLoader)
    f.close()
    net = SpeakerEmbNet(opts)
    print(net)
    #  summary(net.cuda(), (161, 300))
