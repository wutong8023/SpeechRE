import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple


class Conv1dSubsampler(nn.Module):
    """Convolutional subsampler: a stack of 1D convolution (along temporal
    dimension) followed by non-linear activation via gated linear units
    (https://arxiv.org/abs/1911.08460)

    Args:
        in_channels (int): the number of input channels
        mid_channels (int): the number of intermediate channels
        out_channels (int): the number of output channels
        kernel_sizes (List[int]): the kernel size for each convolutional layer
    """

    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        kernel_sizes: List[int] = (3, 3),
        stride: int = 2
    ):
        super(Conv1dSubsampler, self).__init__()
        self.stride = stride
        self.n_layers = len(kernel_sizes)
        self.conv_layers = nn.ModuleList(
            nn.Conv1d(
                in_channels if i == 0 else mid_channels // 2,
                mid_channels if i < self.n_layers - 1 else out_channels * 2,
                k,
                stride,
                padding=k // 2,
            )
            for i, k in enumerate(kernel_sizes)
        )

    def get_out_seq_lens_tensor(self, in_seq_lens_tensor):
        out = in_seq_lens_tensor.clone()
        for _ in range(self.n_layers):
            out = ((out.float() - 1) / self.stride + 1).floor().long()
        return out

    def forward(self, src_tokens, src_lengths):
        bsz, in_seq_len, _ = src_tokens.size()  # B x T x (C x D)
        x = src_tokens.transpose(1, 2).contiguous()  # -> B x (C x D) x T
        for conv in self.conv_layers:
            x = conv(x)
            x = nn.functional.glu(x, dim=1)
            print(x.size())
        _, _, out_seq_len = x.size()
        x = x.transpose(1, 2).transpose(0, 1).contiguous()  # -> T x B x (C x D)
        return x, self.get_out_seq_lens_tensor(src_lengths)

decoder_embed_dim = 512
len_adaptor_channels = 512
len_adaptor_kernel_sizes = "9"

len_adaptor = Conv1dSubsampler(
        decoder_embed_dim,
        len_adaptor_channels,
        decoder_embed_dim,
        [int(k) for k in len_adaptor_kernel_sizes.split(",")],
        stride=8
        )

print(len_adaptor)
input = torch.randn(8, 50, 512)
output = len_adaptor(input, src_lengths=torch.Tensor([50] * 8))
print("Input size:", input.size())
print("Output size:", output[0].size())
print(output[1])