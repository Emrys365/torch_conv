import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.utils import _reverse_repeat_tuple, _single

from .utils import frame, overlap_add


_size_1_t = Union[int, Tuple[int]]


class Conv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_1_t,
        stride: _size_1_t = 1,
        padding: Union[str, _size_1_t] = 0,
        output_padding: Union[str, _size_1_t] = 0,
        dilation: _size_1_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
        transposed: bool = False,
    ):
        """Matrix multiplication based Conv1d and ConvTranspose1d.

        Note:
            ``padding='valid'`` is the same as no padding. ``padding='same'`` pads
            the input so the output has the shape as the input. However, this mode
            doesn't support any stride values other than 1.

        Args:
            in_channels (int): Number of channels in the input image
            out_channels (int): Number of channels produced by the convolution
            kernel_size (int or tuple): Size of the convolving kernel
            stride (int or tuple, optional): Stride of the convolution. Default: 1
            padding (int, tuple or str, optional): Padding added to both sides of
                the input. Default: 0
                When ``transposed`` is ``True``: ``dilation * (kernel_size - 1) - padding``
                zero-padding will be added to both sides of the input. Default: 0
            output_padding (int or tuple, optional): Additional size added to one side of
                the output shape. Only used when ``transposed`` is ``True``. Default: 0
            padding_mode (string, optional): ``'zeros'``, ``'reflect'``,
                ``'replicate'`` or ``'circular'``. Default: ``'zeros'``
            dilation (int or tuple, optional): Spacing between kernel
                elements. Default: 1
            groups (int, optional): Number of blocked connections from input
                channels to output channels. Default: 1
            bias (bool, optional): If ``True``, adds a learnable bias to the
                output. Default: ``True``
            transposed (bool) If ``True``, ConvTranspose1d is used.

        Shape:
            - Input: :math:`(N, C_{in}, L_{in})` or :math:`(C_{in}, L_{in})`
            - Output: :math:`(N, C_{out}, L_{out})` or :math:`(C_{out}, L_{out})`, where

            .. math::
                L_{out} = \left\lfloor\frac{L_{in} + 2 \times \text{padding} - \text{dilation}
                            \times (\text{kernel\_size} - 1) - 1}{\text{stride}} + 1\right\rfloor

        Attributes:
            weight (Tensor): the learnable weights of the module of shape
                :math:`(\text{out\_channels},
                \frac{\text{in\_channels}}{\text{groups}}, \text{kernel\_size})`.
                The values of these weights are sampled from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{groups}{C_\text{in} * \text{kernel\_size}}`
            bias (Tensor):   the learnable bias of the module of shape
                (out_channels). If :attr:`bias` is ``True``, then the values of these weights are
                sampled from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{groups}{C_\text{in} * \text{kernel\_size}}`

        Examples::

            >>> m = Conv1d(16, 33, 3, stride=2)
            >>> input = torch.randn(20, 16, 50)
            >>> output = m(input)
        """
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")
        valid_padding_strings = {"same", "valid"}
        if isinstance(padding, str):
            if padding not in valid_padding_strings:
                raise ValueError(
                    "Invalid padding string {!r}, should be one of {}".format(
                        padding, valid_padding_strings
                    )
                )
            if padding == "same" and any(s != 1 for s in _single(stride)):
                raise ValueError(
                    "padding='same' is not supported for strided convolutions"
                )

        valid_padding_modes = {"zeros", "reflect", "replicate", "circular"}
        if padding_mode not in valid_padding_modes:
            raise ValueError(
                "padding_mode must be one of {}, but got padding_mode='{}'".format(
                    valid_padding_modes, padding_mode
                )
            )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _single(kernel_size)
        self.stride = _single(stride)
        self.padding = padding if isinstance(padding, str) else _single(padding)
        self.dilation = _single(dilation)
        self.groups = groups
        self.padding_mode = padding_mode

        # `_reversed_padding_repeated_twice` is the padding to be passed to
        # `F.pad` if needed (e.g., for non-zero padding types that are
        # implemented as two ops: padding + conv). `F.pad` accepts paddings in
        # reverse order than the dimension.
        if isinstance(self.padding, str):
            self._reversed_padding_repeated_twice = [0, 0] * len(self.kernel_size)
            if padding == "same":
                for d, k, i in zip(
                    self.dilation,
                    self.kernel_size,
                    range(len(self.kernel_size) - 1, -1, -1),
                ):
                    total_padding = d * (k - 1)
                    left_pad = total_padding // 2
                    self._reversed_padding_repeated_twice[2 * i] = left_pad
                    self._reversed_padding_repeated_twice[2 * i + 1] = (
                        total_padding - left_pad
                    )
        else:
            self._reversed_padding_repeated_twice = _reverse_repeat_tuple(
                self.padding, 2
            )

        self.transposed = transposed
        if transposed:
            if isinstance(padding, str):
                raise ValueError(
                    "padding must be an integer or tuple for ConvTranspose1d"
                )
            if padding_mode != "zeros":
                raise ValueError(
                    'Only "zeros" padding mode is supported for ConvTranspose1d'
                )
            self.output_padding = _single(output_padding)
            self.weight = nn.Parameter(
                torch.empty(
                    (in_channels, out_channels // groups, *self.kernel_size),
                    **factory_kwargs
                )
            )
        else:
            self.output_padding = _single(0)
            self.weight = nn.Parameter(
                torch.empty(
                    (out_channels, in_channels // groups, *self.kernel_size),
                    **factory_kwargs
                )
            )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(k), 1/sqrt(k)), where k = weight.size(1) * prod(*kernel_size)
        # For more details see: https://github.com/pytorch/pytorch/issues/15314#issuecomment-477448573
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)

    def extra_repr(self):
        s = (
            "{in_channels}, {out_channels}, kernel_size={kernel_size}"
            ", stride={stride}"
        )
        if self.padding != (0,) * len(self.padding):
            s += ", padding={padding}"
        if self.dilation != (1,) * len(self.dilation):
            s += ", dilation={dilation}"
        if self.output_padding != (0,) * len(self.output_padding):
            s += ", output_padding={output_padding}"
        if self.groups != 1:
            s += ", groups={groups}"
        if self.bias is None:
            s += ", bias=False"
        if self.padding_mode != "zeros":
            s += ", padding_mode={padding_mode}"
        if self.transposed:
            s += ", transposed={transposed}"
        return s.format(**self.__dict__)

    def _conv_forward(self, input: Tensor):
        if self.padding_mode != "zeros":
            return F.conv1d(
                F.pad(
                    input, self._reversed_padding_repeated_twice, mode=self.padding_mode
                ),
                self.weight,
                self.bias,
                self.stride,
                _single(0),
                self.dilation,
                self.groups,
            )
        return F.conv1d(
            input,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

    def _conv_transposed_forward(
        self, input: Tensor, output_size: Optional[List[int]] = None
    ):
        if self.padding_mode != "zeros":
            raise ValueError(
                "Only `zeros` padding mode is supported for ConvTranspose1d"
            )

        assert isinstance(self.padding, tuple)
        # One cannot replace List by Tuple or Sequence in "_output_padding" because
        # TorchScript does not support `Sequence[T]` or `Tuple[T, ...]`.
        num_spatial_dims = 1
        output_padding = self._output_padding(
            input,
            output_size,
            self.stride,
            self.padding,
            self.kernel_size,
            num_spatial_dims,
            self.dilation,
        )
        return F.conv_transpose1d(
            input,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            output_padding,
            self.groups,
            self.dilation,
        )

    def _output_padding(
        self,
        input: Tensor,
        output_size: Optional[List[int]],
        stride: List[int],
        padding: List[int],
        kernel_size: List[int],
        num_spatial_dims: int,
        dilation: Optional[List[int]] = None,
    ) -> List[int]:
        # copied from torch.nn.modules.conv._ConvTransposeNd._output_padding
        if output_size is None:
            ret = _single(self.output_padding)  # converting to list if was not already
        else:
            has_batch_dim = input.dim() == num_spatial_dims + 2
            num_non_spatial_dims = 2 if has_batch_dim else 1
            if len(output_size) == num_non_spatial_dims + num_spatial_dims:
                output_size = output_size[num_non_spatial_dims:]
            if len(output_size) != num_spatial_dims:
                raise ValueError(
                    "ConvTranspose{}D: for {}D input, output_size must have {} or {} elements (got {})".format(
                        num_spatial_dims,
                        input.dim(),
                        num_spatial_dims,
                        num_non_spatial_dims + num_spatial_dims,
                        len(output_size),
                    )
                )

            min_sizes = torch.jit.annotate(List[int], [])
            max_sizes = torch.jit.annotate(List[int], [])
            for d in range(num_spatial_dims):
                dim_size = (
                    (input.size(d + num_non_spatial_dims) - 1) * stride[d]
                    - 2 * padding[d]
                    + (dilation[d] if dilation is not None else 1)
                    * (kernel_size[d] - 1)
                    + 1
                )
                min_sizes.append(dim_size)
                max_sizes.append(min_sizes[d] + stride[d] - 1)

            for i in range(len(output_size)):
                size = output_size[i]
                min_size = min_sizes[i]
                max_size = max_sizes[i]
                if size < min_size or size > max_size:
                    raise ValueError(
                        (
                            "requested an output size of {}, but valid sizes range "
                            "from {} to {} (for an input of {})"
                        ).format(output_size, min_sizes, max_sizes, input.size()[2:])
                    )

            res = torch.jit.annotate(List[int], [])
            for d in range(num_spatial_dims):
                res.append(output_size[d] - min_sizes[d])

            ret = res
        return ret

    def pad(self, input: Tensor) -> Tensor:
        if self.padding_mode == "zeros":
            return F.pad(
                input, self._reversed_padding_repeated_twice, mode="constant", value=0
            )
        return F.pad(
            input, self._reversed_padding_repeated_twice, mode=self.padding_mode
        )

    def _conv_forward_matmul(self, input: Tensor):
        input2 = self.pad(input)
        # Window the time series (B, in_channels, T) -> (B, in_channels, kernel_size, T')
        frames = frame(
            input2,
            kernel_size=self.kernel_size[0],
            stride=self.stride[0],
            dilation=self.dilation[0],
            dim=-1,
        ).contiguous()

        if self.groups > 1:
            B, in_ch, ks, T = frames.shape
            frames = frames.reshape(B, self.groups, -1, ks, T)
            weight = self.weight.reshape(self.groups, -1, frames.size(2), ks)
            conv_matrix = torch.einsum("goik,bgikt->bgot", weight, frames)
            conv_matrix = conv_matrix.view(B, -1, T).contiguous()
        else:
            conv_matrix = torch.einsum("oik,bikt->bot", self.weight, frames)
        if self.bias is not None:
            conv_matrix = conv_matrix + self.bias.view(1, -1, 1)
        return conv_matrix

    def _conv_transposed_forward_matmul(
        self, input: Tensor, output_size: Optional[List[int]] = None
    ):
        if self.padding_mode != "zeros":
            raise ValueError(
                "Only `zeros` padding mode is supported for ConvTranspose1d"
            )

        assert isinstance(self.padding, tuple)
        # One cannot replace List by Tuple or Sequence in "_output_padding" because
        # TorchScript does not support `Sequence[T]` or `Tuple[T, ...]`.
        num_spatial_dims = 1
        output_padding = self._output_padding(
            input,
            output_size,
            self.stride,
            self.padding,
            self.kernel_size,
            num_spatial_dims,
            self.dilation,
        )

        B, in_ch, T = input.shape
        ks = self.weight.size(2)
        if self.groups > 1:
            input = input.reshape(B, self.groups, -1, T)
            weight = self.weight.reshape(self.groups, input.size(2), -1, ks)
            convT_matrix = torch.einsum("giok,bgit->bgokt", weight, input)
        else:
            convT_matrix = torch.einsum("iok,bit->bokt", self.weight, input)
        convT_matrix = overlap_add(
            convT_matrix.reshape(-1, ks, T), self.stride[0], dilation=self.dilation[0]
        )
        convT_matrix = convT_matrix.view(B, -1, convT_matrix.size(1)).contiguous()
        convT_matrix = F.pad(convT_matrix, (0, output_padding[0]))
        if self.padding[0] > 0:
            convT_matrix = convT_matrix[..., self.padding[0] : -self.padding[0]]
        if self.bias is not None:
            convT_matrix = convT_matrix + self.bias.view(1, -1, 1)
        return convT_matrix

    def forward(self, input: Tensor, output_size: Optional[List[int]] = None) -> Tensor:
        if self.transposed:
            return self._conv_transposed_forward_matmul(input, output_size)
            # return self._conv_transposed_forward(input, output_size)
        else:
            return self._conv_forward_matmul(input)
            # return self._conv_forward(input)


if __name__ == "__main__":
    for (in_ch, out_ch, ks, stride, pad, dilation, groups, bias, pad_mode) in [
        (1, 3, 3, 1, 2, 1, 1, True, "zeros"),
        (1, 3, 3, 1, "same", 1, 1, True, "zeros"),
        (1, 3, 3, 1, "same", 1, 1, False, "zeros"),
        (2, 4, 3, 1, "same", 1, 2, True, "zeros"),
        (1, 3, 3, 1, "same", 1, 1, True, "circular"),
        (1, 3, 3, 1, "same", 1, 1, True, "reflect"),
        (1, 3, 3, 1, "same", 1, 1, True, "replicate"),
        (1, 3, 3, 2, "valid", 2, 1, True, "zeros"),
    ]:
        opt = dict(
            stride=stride,
            padding=pad,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=pad_mode,
        )
        conv_th = nn.Conv1d(in_ch, out_ch, ks, **opt)
        conv = Conv1d(in_ch, out_ch, ks, **opt)
        conv.weight = conv_th.weight
        conv.bias = conv_th.bias
        length = 42
        x = torch.rand(2, in_ch, length)
        torch.testing.assert_close(conv_th(x), conv(x))

        if pad_mode != "zeros":
            continue
        if isinstance(pad, str):
            opt["padding"] = 0 if pad == "valid" else (ks - 1) * dilation // 2
        conv_transpose_th = nn.ConvTranspose1d(out_ch, in_ch, ks, **opt)
        conv_transpose = Conv1d(out_ch, in_ch, ks, transposed=True, **opt)
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
