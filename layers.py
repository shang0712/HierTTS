# -*- coding: utf-8 -*-

# Copyright 2020 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

"""Pseudo QMF modules."""

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from scipy.signal import kaiser

class RepConv(nn.Module):
    def __init__(self, channels, kernel_sizes, dilation=3):
        super(RepConv, self).__init__()

        self.channels = channels
        self.dilation = dilation
        self.kernel_sizes = kernel_sizes
        self.max_kernel_size = self.kernel_sizes
        self.padding = (self.max_kernel_size - 1) // 2 * dilation

        self.convs = nn.ModuleList()
        #for k in self.kernel_sizes:
        self.convs.append(nn.Conv1d(channels, channels, kernel_sizes, dilation=dilation, padding=(kernel_sizes - 1) // 2 * dilation))
        self.convs.append(nn.Conv1d(channels, channels, kernel_sizes, dilation=1, padding=(kernel_sizes - 1) // 2 * 1))

    def forward(self, x):
        conv_outputs = []
        for conv in self.convs:
            conv_outputs.append(conv(x))
        return x + sum(conv_outputs) / len(self.convs)

    def inference(self, x):
        if not hasattr(self, 'weight') or not hasattr(self, 'bias'):
            raise ValueError('do not has such attribute, please call _convert_weight_and_bias first!')

        return F.conv1d(x, self.weight.to(x.device), self.bias.to(x.device), dilation=self.dilation,
                        padding=self.padding)

    def convert_weight_bias(self):
        weight = self.convs[-1].weight
        bias = self.convs[-1].bias

        # add other conv
        for conv in self.convs[:-1]:
            pad = (self.max_kernel_size - conv.weight.shape[-1]) // 2
            weight = weight + F.pad(conv.weight, [pad, pad])
            bias = bias + conv.bias

        weight = weight / len(self.convs)
        bias = bias / len(self.convs)

        # add identity
        pad = (self.max_kernel_size - 1) // 2
        weight = weight + F.pad(torch.eye(self.channels).unsqueeze(-1).to(weight.device), [pad, pad])

        self.weight = weight.detach()
        self.bias = bias.detach()

def design_prototype_filter(taps=62, cutoff_ratio=0.14204204204204204, beta=9.0):
    """Design prototype filter for PQMF.

    This method is based on `A Kaiser window approach for the design of prototype
    filters of cosine modulated filterbanks`_.

    Args:
        taps (int): The number of filter taps.
        cutoff_ratio (float): Cut-off frequency ratio.
        beta (float): Beta coefficient for kaiser window.

    Returns:
        ndarray: Impluse response of prototype filter (taps + 1,).

    .. _`A Kaiser window approach for the design of prototype filters of cosine modulated filterbanks`:
        https://ieeexplore.ieee.org/abstract/document/681427

    """
    # check the arguments are valid
    assert taps % 2 == 0, "The number of taps mush be even number."
    assert 0.0 < cutoff_ratio < 1.0, "Cutoff ratio must be > 0.0 and < 1.0."

    # make initial filter
    omega_c = np.pi * cutoff_ratio
    with np.errstate(invalid='ignore'):
        h_i = np.sin(omega_c * (np.arange(taps + 1) - 0.5 * taps)) \
            / (np.pi * (np.arange(taps + 1) - 0.5 * taps))
    h_i[taps // 2] = np.cos(0) * cutoff_ratio  # fix nan due to indeterminate form

    # apply kaiser window
    w = kaiser(taps + 1, beta)
    h = h_i * w

    return h


class PQMF(torch.nn.Module):
    """PQMF module.

    This module is based on `Near-perfect-reconstruction pseudo-QMF banks`_.

    .. _`Near-perfect-reconstruction pseudo-QMF banks`:
        https://ieeexplore.ieee.org/document/258122

    """

    def __init__(self, subbands=4, taps=62, cutoff_ratio=0.14204204204204204, beta=9.0):
        """Initilize PQMF module.

        The cutoff_ratio and beta parameters are optimized for #subbands = 4.
        See dicussion in https://github.com/kan-bayashi/ParallelWaveGAN/issues/195.

        Args:
            subbands (int): The number of subbands.
            taps (int): The number of filter taps.
            cutoff_ratio (float): Cut-off frequency ratio.
            beta (float): Beta coefficient for kaiser window.

        """
        super(PQMF, self).__init__()

        # build analysis & synthesis filter coefficients
        h_proto = design_prototype_filter(taps, cutoff_ratio, beta)
        h_analysis = np.zeros((subbands, len(h_proto)))
        h_synthesis = np.zeros((subbands, len(h_proto)))
        for k in range(subbands):
            h_analysis[k] = 2 * h_proto * np.cos(
                (2 * k + 1) * (np.pi / (2 * subbands)) *
                (np.arange(taps + 1) - (taps / 2)) +
                (-1) ** k * np.pi / 4)
            h_synthesis[k] = 2 * h_proto * np.cos(
                (2 * k + 1) * (np.pi / (2 * subbands)) *
                (np.arange(taps + 1) - (taps / 2)) -
                (-1) ** k * np.pi / 4)

        # convert to tensor
        analysis_filter = torch.from_numpy(h_analysis).float().unsqueeze(1)
        synthesis_filter = torch.from_numpy(h_synthesis).float().unsqueeze(0)

        # register coefficients as beffer
        self.register_buffer("analysis_filter", analysis_filter)
        #self.register_buffer("synthesis_filter", synthesis_filter)
        self.synthesis_filter = torch.nn.parameter.Parameter(synthesis_filter, requires_grad=True)
        # filter for downsampling & upsampling
        updown_filter = torch.zeros((subbands, subbands, subbands)).float()
        for k in range(subbands):
            updown_filter[k, k, 0] = 1.0
        self.register_buffer("updown_filter", updown_filter)
        self.subbands = subbands

        # keep padding info
        self.pad_fn = torch.nn.ConstantPad1d(taps // 2, 0.0)

    def analysis(self, x):
        """Analysis with PQMF.

        Args:
            x (Tensor): Input tensor (B, 1, T).

        Returns:
            Tensor: Output tensor (B, subbands, T // subbands).

        """
        x = F.conv1d(self.pad_fn(x), self.analysis_filter)
        return F.conv1d(x, self.updown_filter, stride=self.subbands)

    def synthesis(self, x):
        """Synthesis with PQMF.

        Args:
            x (Tensor): Input tensor (B, subbands, T // subbands).

        Returns:
            Tensor: Output tensor (B, 1, T).

        """
        # NOTE(kan-bayashi): Power will be dreased so here multipy by # subbands.
        #   Not sure this is the correct way, it is better to check again.
        # TODO(kan-bayashi): Understand the reconstruction procedure
        x = F.conv_transpose1d(x, self.updown_filter * self.subbands, stride=self.subbands)
        return F.conv1d(self.pad_fn(x), self.synthesis_filter)
