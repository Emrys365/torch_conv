import time
import warnings

import numpy as np
import pytest
import torch

from torch_conv import Conv1d


is_cuda_available = torch.cuda.is_available()


def test_speed(func, *args, num_runs=100, **kwargs):
    ret = []
    for _ in range(num_runs):
        start = time.perf_counter()
        func(*args, **kwargs)
        end = time.perf_counter()
        ret.append((end - start) * 1000)

    mean, std = np.mean(ret), np.std(ret)
    info = "{:.1f} ms ± {:.1f} ms per loop (mean ± std. dev. of {} runs, 1 loop each"
    print(info.format(mean, std, num_runs))
    return mean, std


@pytest.mark.parametrize(
    "in_channels, out_channels, groups", [(1, 128, 1), (2, 256, 2)]
)
@pytest.mark.parametrize(
    "kernel_size, stride, dilation", [(16, 8, 1), (2, 1, 1), (256, 128, 2)]
)
@pytest.mark.parametrize("padding", ["same", "valid", 1])
@pytest.mark.parametrize("pad_mode", ["zeros", "reflect", "circular", "replicate"])
@pytest.mark.parametrize("num_runs", [100])
def test_Conv1d_speed(
    in_channels,
    out_channels,
    kernel_size,
    stride,
    padding,
    dilation,
    groups,
    pad_mode,
    num_runs,
):
    if stride != 1 and padding == "same":
        return

    elapsed_time = {}
    x = torch.randn(1, in_channels, 16000)
    opt = dict(
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=True,
        padding_mode=pad_mode,
    )
    conv = Conv1d(in_channels, out_channels, kernel_size, **opt)
    if is_cuda_available:
        conv.cuda()
        x = x.cuda()
    elapsed_time["impl"] = test_speed(conv, x, num_runs=num_runs)[0]

    conv_th = torch.nn.Conv1d(in_channels, out_channels, kernel_size, **opt)
    elapsed_time["native"] = test_speed(conv_th, x, num_runs=num_runs)[0]
    if elapsed_time["impl"] < elapsed_time["native"]:
        info = (
            "Our implementation ({:.4f} ms) is faster than " "`torch.stft` ({:.1f} ms)"
        )
        warnings.warn(info.format(elapsed_time["impl"], elapsed_time["native"]))


@pytest.mark.parametrize(
    "in_channels, out_channels, groups", [(1, 128, 1), (2, 256, 2)]
)
@pytest.mark.parametrize(
    "kernel_size, stride, dilation", [(16, 8, 1), (2, 1, 1), (256, 128, 2)]
)
@pytest.mark.parametrize("padding", ["same", "valid", 1])
@pytest.mark.parametrize("pad_mode", ["zeros"])
@pytest.mark.parametrize("num_runs", [100])
def test_ConvTranspose1d_speed(
    in_channels,
    out_channels,
    kernel_size,
    stride,
    padding,
    dilation,
    groups,
    pad_mode,
    num_runs,
):
    elapsed_time = {}
    x = torch.randn(1, out_channels, 800)
    opt = dict(
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=True,
        padding_mode=pad_mode,
    )
    if isinstance(padding, str):
        opt["padding"] = 0 if padding == "valid" else (kernel_size - 1) * dilation // 2
    conv = Conv1d(out_channels, in_channels, kernel_size, transposed=True, **opt)
    if is_cuda_available:
        conv.cuda()
        x = x.cuda()
    elapsed_time["impl"] = test_speed(conv, x, num_runs=num_runs)[0]

    conv_th = torch.nn.ConvTranspose1d(out_channels, in_channels, kernel_size, **opt)
    elapsed_time["native"] = test_speed(conv_th, x, num_runs=num_runs)[0]
    if elapsed_time["impl"] < elapsed_time["native"]:
        info = (
            "Our implementation ({:.4f} ms) is faster than " "`torch.stft` ({:.1f} ms)"
        )
        warnings.warn(info.format(elapsed_time["impl"], elapsed_time["native"]))
