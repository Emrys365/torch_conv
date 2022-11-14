import torch


def frame(x, kernel_size, stride, dilation=1, dim=-1):
    # import librosa
    # import numpy as np
    # np.testing.assert_allclose(
    #     frame(x, frame_length, hop_length, dim=-1).numpy(),
    #     librosa.util.frame(x.numpy(), frame_length, hop_length, axis=-1),
    # )
    out_length = x.size(dim) - dilation * (kernel_size - 1) - 1
    if out_length <= 0:
        raise ValueError(
            "Input is too short (n={:d})"
            " for kernel_size={:d} and dilation={:d}".format(
                x.size(dim), kernel_size, dilation
            )
        )

    if stride < 1:
        raise ValueError("Invalid stride: {:d}".format(stride))

    n_frames = out_length // stride + 1
    strides = list(x.stride())

    if dim == -1 or dim == x.dim() - 1:
        shape = list(x.shape[:-1]) + [kernel_size, n_frames]
        strides = strides[:-1] + [strides[-1] * dilation, stride]

    elif dim == 0:
        shape = [n_frames, kernel_size] + list(x.shape)[1:]
        strides = [stride * strides[0], dilation * strides[0]] + strides[1:]

    else:
        raise ValueError("Frame dim={} must be either 0 or -1".format(dim))

    return x.as_strided(shape, strides, storage_offset=0)


def overlap_add(seq, stride, dilation=1, mode="sum"):
    """Perform Overlap-Add on the input sequence `seq` with overlapped parts averaged.

    Args:
        seq (torch.Tensor): shape (B, T, num_segments)
        stride (int): hop size between adjacent segments
        dilation (int): dilation in each segment
        mode (str): one of ("sum", "average")
            - "sum": overlapped parts will be summed
            - "average": overlapped parts will be averaged (allowing > 2 overlapping folds)

    Returns:
        seq_fold (torch.Tensor): shape (B, out_samples)
    """
    assert mode in ("sum", "average"), f"Unsupported mode: {mode}"
    B, kernel_size, num_segments = seq.shape
    out_samples = (num_segments - 1) * stride + dilation * (kernel_size - 1) + 1
    if mode == "sum":
        seq_fold = torch.nn.functional.fold(
            input=seq,
            output_size=(1, out_samples),
            kernel_size=(1, kernel_size),
            dilation=(1, dilation),
            stride=(1, stride),
        )
    elif mode == "average":
        seq_fold = torch.nn.functional.fold(
            input=seq,
            output_size=(1, out_samples),
            kernel_size=(1, kernel_size),
            dilation=(1, dilation),
            stride=(1, stride),
        )
        norm_mat = torch.nn.functional.fold(
            input=torch.ones_like(seq),
            output_size=(1, out_samples),
            kernel_size=(1, kernel_size),
            dilation=(1, dilation),
            stride=(1, stride),
        )
        seq_fold = seq_fold / norm_mat
    return seq_fold.squeeze(2).squeeze(1)
