import pytest

import torch
import torch.nn as nn
from torch_conv import Conv1d


@pytest.mark.parametrize(
    "in_channels, out_channels, groups", [(1, 2, 1), (3, 2, 1), (2, 4, 2), (6, 2, 2)]
)
@pytest.mark.parametrize("kernel_size, stride", [(16, 8), (2, 1)])
@pytest.mark.parametrize("dilation", [1, 2, 3])
@pytest.mark.parametrize("padding", ["same", "valid", 0, 1])
@pytest.mark.parametrize("pad_mode", ["zeros", "reflect", "circular", "replicate"])
@pytest.mark.parametrize("bias", [True, False])
def test_Conv1d_consistency(
    in_channels,
    out_channels,
    kernel_size,
    stride,
    padding,
    dilation,
    groups,
    bias,
    pad_mode,
):
    if stride != 1 and padding == "same":
        return

    x = torch.randn(2, in_channels, 800)
    opt = dict(
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=bias,
        padding_mode=pad_mode,
    )
    conv_th = nn.Conv1d(in_channels, out_channels, kernel_size, **opt)
    conv = Conv1d(in_channels, out_channels, kernel_size, **opt)
    conv.weight = conv_th.weight
    conv.bias = conv_th.bias
    torch.testing.assert_close(conv_th(x), conv(x))


@pytest.mark.parametrize(
    "in_channels, out_channels, groups", [(1, 2, 1), (3, 2, 1), (2, 4, 2), (6, 2, 2)]
)
@pytest.mark.parametrize("kernel_size, stride", [(16, 8), (4, 2)])
@pytest.mark.parametrize("dilation", [1, 2, 3])
@pytest.mark.parametrize("padding", ["same", "valid", 0, 1])
@pytest.mark.parametrize("pad_mode", ["zeros"])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("output_padding", [0, 1, 3])
def test_ConvTranspose1d_consistency(
    in_channels,
    out_channels,
    kernel_size,
    stride,
    padding,
    dilation,
    groups,
    bias,
    pad_mode,
    output_padding,
):
    if stride != 1 and padding == "same":
        return

    length = 800
    x = torch.randn(2, in_channels, length)
    opt = dict(
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=bias,
        padding_mode=pad_mode,
    )
    conv_th = nn.Conv1d(in_channels, out_channels, kernel_size, **opt)
    conv = Conv1d(in_channels, out_channels, kernel_size, **opt)
    conv.weight = conv_th.weight
    conv.bias = conv_th.bias

    if output_padding >= stride and output_padding >= dilation:
        return
    opt["output_padding"] = output_padding
    if isinstance(padding, str):
        opt["padding"] = 0 if padding == "valid" else (kernel_size - 1) * dilation // 2
    conv_transpose_th = nn.ConvTranspose1d(
        out_channels, in_channels, kernel_size, **opt
    )
    conv_transpose = Conv1d(
        out_channels, in_channels, kernel_size, transposed=True, **opt
    )
    conv_transpose.weight = conv_transpose_th.weight
    conv_transpose.bias = conv_transpose_th.bias
    torch.testing.assert_close(
        conv_transpose_th(conv_th(x), output_size=None),
        conv_transpose(conv(x), output_size=None),
    )
    torch.testing.assert_close(
        conv_transpose_th(conv_th(x), output_size=(length,)),
        conv_transpose(conv(x), output_size=(length,)),
    )
