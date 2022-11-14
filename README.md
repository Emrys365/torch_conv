# PyTorch-based implementations of Conv1d and ConvTranspose1d

This repository provides purely PyTorch-based Conv1d and ConvTranspose1d implementations.

## Install

```bash
# install via git
python -m pip install git+https://github.com/Emrys365/torch_conv

# install from source
git clone git@github.com:Emrys365/torch_conv.git
cd torch_conv
python -m pip install -e .
```

## Usage

```python
import torch
from torch_conv import Conv1d

device = "cpu"
kernel_size = 256
dilation = 1
padding = (kernel_size - 1) * dilation // 2
opt = dict(
    stride=128,
    padding=padding,
    dilation=dilation,
    groups=2,
    bias=True,
    padding_mode="zeros",
)
conv_th = torch.nn.Conv1d(6, 2, kernel_size, device=device, **opt)
conv = Conv1d(6, 2, kernel_size, device=device, **opt)
conv.weight = conv_th.weight
conv.bias = conv_th.bias

signal = torch.rand(2, 6, 8000, device=device)
spec = conv(signal)
spec_th = conv_th(signal)
assert torch.allclose(spec, spec_th)

conv_transpose_th = torch.nn.ConvTranspose1d(2, 6, kernel_size, device=device, **opt)
conv_transpose = Conv1d(2, 6, kernel_size, device=device, transposed=True, **opt)
conv_transpose.weight = conv_transpose_th.weight
conv_transpose.bias = conv_transpose_th.bias
signal_dec = conv_transpose(spec)
signal_dec_th = conv_transpose_th(spec_th)
assert torch.allclose(signal_dec, signal_dec_th)
```

## Test implementations

```bash
python -m pytest tests/
```
