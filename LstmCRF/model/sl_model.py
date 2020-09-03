# -*- coding: utf-8 -*-
# @Time    : 2020/7/20 10:47
# @Author  : piguanghua
# @FileName: sl_model.py
# @Software: PyCharm

import torch
import torch.nn as nn
import torch.optim as optim
from model.torchcrf import CRF
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from model.functional.initialize import init_embedding

class CharFeature(nn.Module):

    def __init__(self, **kwargs):
        """
        Args:
            feature_size: int, 字符表的大小
            feature_dim: int, 字符embedding 维度
            require_grad: bool，char的embedding表是否需要更新

            filter_sizes: list(int), 卷积核尺寸
            filter_nums: list(int), 卷积核数量
        """
        super(CharFeature, self).__init__()
        for k in kwargs:
            self.__setattr__(k, kwargs[k])

        # char embedding layer
        self.char_embedding = nn.Embedding(self.feature_size, self.feature_dim)
        init_embedding(self.char_embedding.weight)

        # cnn
        self.char_encoders = nn.ModuleList()
        for i, filter_size in enumerate(self.filter_sizes):
            f = nn.Conv3d(
                in_channels=1, out_channels=self.filter_nums[i], kernel_size=(1, filter_size, self.feature_dim))
            self.char_encoders.append(f)

    def forward(self, inputs):
        """
        Args:
            inputs: 3D tensor, [bs, max_len, max_len_char]

        Returns:
            char_conv_outputs: 3D tensor, [bs, max_len, output_dim]
        """
        max_len, max_len_char = inputs.size(1), inputs.size(2)
        inputs = inputs.view(-1, max_len * max_len_char)  # [bs, -1]
        input_embed = self.char_embedding(inputs)  # [bs, ml*ml_c, feature_dim]
        # [bs, 1, max_len, max_len_char, feature_dim]
        input_embed = input_embed.view(-1, 1, max_len, max_len_char, self.feature_dim)

        # conv
        char_conv_outputs = []
        for char_encoder in self.char_encoders:
            conv_output = char_encoder(input_embed)
            pool_output = torch.squeeze(torch.max(conv_output, -2)[0])
            char_conv_outputs.append(pool_output)
        char_conv_outputs = torch.cat(char_conv_outputs, dim=1)

        # size=[bs, max_len, output_dim]
        char_conv_outputs = char_conv_outputs.transpose(-2, -1).contiguous()

        return char_conv_outputs

class SLModel(nn.Module):
    def __init__(self, config, char2id, tag2id, emb_matrix, device):
        super(SLModel, self).__init__()

        self.hidden_dim = config.hidden_dim
        self.vocab_size = len(char2id)
        self.seg_size = 5
        self.tag_to_ix = tag2id
        self.tagset_size = len(tag2id)

        """ pdding_idx=0，也就是让pad标记不更新 """
        self.char_emb = nn.Embedding.from_pretrained(
            emb_matrix, freeze=False, padding_idx=0
        )

