import torch
import torch.nn.functional as F
import numpy as np
from torch.autograd import Function
import pywt


def mode_to_int(mode):
    if mode == 'zero':
        return 0
    elif mode == 'symmetric':
        return 1
    elif mode == 'per' or mode == 'periodization':
        return 2
    elif mode == 'constant':
        return 3
    elif mode == 'reflect':
        return 4
    elif mode == 'replicate':
        return 5
    elif mode == 'periodic':
        return 6
    else:
        raise ValueError("Unkown pad type: {}".format(mode))


def int_to_mode(mode):
    if mode == 0:
        return 'zero'
    elif mode == 1:
        return 'symmetric'
    elif mode == 2:
        return 'periodization'
    elif mode == 3:
        return 'constant'
    elif mode == 4:
        return 'reflect'
    elif mode == 5:
        return 'replicate'
    elif mode == 6:
        return 'periodic'
    else:
        raise ValueError("Unkown pad type: {}".format(mode))


def prep_filt_afb1d(h0, h1, device=None):
    """
    Prepares the filters to be of the right form for the afb3d function.  In
    particular, makes the tensors the right shape. It takes mirror images of
    them as as afb3d uses conv3d which acts like normal correlation.

    Inputs:
        h0 (array-like): low pass column filter bank
        h1 (array-like): high pass column filter bank
        device: which device to put the tensors on to

    Returns:
        (h0, h1)
    """
    h0 = np.array(h0[::-1]).ravel()
    h1 = np.array(h1[::-1]).ravel()
    t = torch.get_default_dtype()
    h0 = torch.tensor(h0, device=device, dtype=t).reshape((1, 1, -1))
    h1 = torch.tensor(h1, device=device, dtype=t).reshape((1, 1, -1))
    return h0, h1


def afb1d(x, h0, h1, mode='zero', dim=-1):
    """ 1D analysis filter bank (along one dimension only) of a tensor

    Inputs:
        x (tensor): 5D input with the last three dimensions the spatial input
        h0 (tensor): 5D input for the lowpass filter. Should have shape (1, 1, d, 1, 1)
            or (1, 1, 1, h, 1) or (1, 1, 1, 1, w)
        h1 (tensor): 5D input for the highpass filter. Should have shape (1, 1, d, 1, 1)
            or (1, 1, 1, h, 1) or (1, 1, 1, 1, w)
        mode (str): padding method
        dim (int) - dimension of filtering. d=2 is for a temporal filter (called
            tube filtering but filters across the depth). d=3 is for a
            vertical filter (called column filtering but filters across the rows) and d=4
            is for a horizontal filter, (called row filtering but filters across the
            columns).

    Returns:
        lohi: lowpass and highpass subbands concatenated along the channel
            dimension
    """
    num_channels = x.shape[1]  # Number of channels
    d = dim % 5                # Convert the dim to positive
    N = x.shape[d]             # Length of signal
    assert isinstance(h0, torch.Tensor) and isinstance(h1, torch.Tensor), "Either h0 or h1 is not of type torch.Tensor"

    L = h0.numel()             # Length of filter coefficients
    shape = [1, 1, 1, 1, 1]
    shape[d] = L
    assert h0.shape == h1.shape == tuple(shape), f"Wrong filter shapes. Expected {shape} but" \
                                                 f" got h0{h0.shape}, h1{h1.shape}"
    h = torch.cat([h0, h1] * num_channels, dim=0)

    # Calculate the pad size
    outsize = pywt.dwt_coeff_len(N, L, mode=mode)
    p = 2 * (outsize - 1) - N + L
    assert p % 2 == 0, f"Assuming even length signals to avoid pre-padding."
    if d == 2:
        stride = (2, 1, 1)
        padding = (p // 2, 0, 0)
    elif d == 3:
        stride = (1, 2, 1)
        padding = (0, p // 2, 0)
    elif d == 4:
        stride = (1, 1, 2)
        padding = (0, 0, p // 2)
    else:
        raise ValueError(f"afb1d in lowlevel_3d is expected to be called with dim as 2, 3 or 4. Got {d}")

    if mode == 'zero':
        # Calculate the high and lowpass
        lohi = F.conv3d(x, h, padding=padding, stride=stride, groups=num_channels)
    else:
        raise ValueError("Unkown pad type: {} for afb1d in lowlevel_3d".format(mode))

    return lohi


def prep_filt_sfb1d(g0, g1, device=None):
    """
    Prepares the filters to be of the right form for the sfb1d function. In
    particular, makes the tensors the right shape. It does not mirror image them
    as as sfb1d uses conv3d_transpose which acts like normal convolution.

    Inputs:
        g0 (array-like): low pass filter bank
        g1 (array-like): high pass filter bank
        device: which device to put the tensors on to

    Returns:
        (g0, g1)
    """
    g0 = np.array(g0).ravel()
    g1 = np.array(g1).ravel()
    t = torch.get_default_dtype()
    g0 = torch.tensor(g0, device=device, dtype=t).reshape((1, 1, -1))
    g1 = torch.tensor(g1, device=device, dtype=t).reshape((1, 1, -1))

    return g0, g1


def sfb1d(lo, hi, g0, g1, mode='zero', dim=-1):
    """ 1D synthesis filter bank (along one dimension only) of a tensor

    Inputs:
        lo (tensor): 5D tensor containing approximation coefficients (N, C, D, H, W)
        hi (tensor): 5D tensor containing detail coefficients (N, C, D, H, W)
        g0 (tensor): 5D input for the lowpass filter. Should have shape (1, 1, d, 1, 1)
            or (1, 1, 1, h, 1) or (1, 1, 1, 1, w)
        g1 (tensor): 5D input for the highpass filter. Should have shape (1, 1, d, 1, 1)
            or (1, 1, 1, h, 1) or (1, 1, 1, 1, w)
        mode (str): padding method
        dim (int) - dimension of filtering. d=2 is for a temporal filter (called
            tube filtering but filters across the depth). d=3 is for a
            vertical filter (called column filtering but filters across the rows) and d=4
            is for a horizontal filter, (called row filtering but filters across the
            columns).

    Returns:
        x: reconstructed 5D tensor (N, C, D, H, W)
    """
    num_channels = lo.shape[1] # Number of channels
    d = dim % 5                # Convert the dim to positive
    assert isinstance(g0, torch.Tensor) and isinstance(g1, torch.Tensor), "Either g0 or g1 is not of type torch.Tensor"
    L = g0.numel()  # Length of filter coefficients
    shape = [1, 1, 1, 1, 1]
    shape[d] = L
    assert g0.shape == g1.shape == tuple(shape), f"Wrong filter shapes. Expected {shape} but" \
                                                 f" got g0{g0.shape}, g1{g1.shape}"
    N = 2 * lo.shape[d]         # Length of output signal
    g0 = torch.cat([g0] * num_channels, dim=0)
    g1 = torch.cat([g1] * num_channels, dim=0)
    if d == 2:
        stride = (2, 1, 1)
        padding = (L-2, 0, 0)
    elif d == 3:
        stride = (1, 2, 1)
        padding = (0, L-2, 0)
    elif d == 4:
        stride = (1, 1, 2)
        padding = (0, 0, L-2)
    else:
        raise ValueError(f"afb1d in lowlevel_3d is expected to be called with dim as 2, 3 or 4. Got {d}")

    if mode == 'zero' or mode == 'symmetric' or mode == 'reflect' or mode == 'periodic':
        y = F.conv_transpose3d(lo, g0, stride=stride, padding=padding, groups=num_channels) + \
            F.conv_transpose3d(hi, g1, stride=stride, padding=padding, groups=num_channels)
    else:
        raise ValueError("Unkown pad type: {}".format(mode))

    return y


def prep_filt_afb3d(h0_tube, h1_tube, h0_col=None, h1_col=None, h0_row=None, h1_row=None, device=None):
    """
    Prepares the filters to be of the right form for the afb3d function.  In
    particular, makes the tensors the right shape. It takes mirror images of
    them as as afb3d uses conv3d which acts like normal correlation.

    Inputs:
        h0_tube (array-like): low pass tube filter bank
        h1_tube (array-like): high pass tube filter bank
        h0_col (array-like): low pass column filter bank. If none, will assume the
            same as tube filter
        h1_col (array-like): high pass column filter bank. If none, will assume the
            same as tube filter
        h0_row (array-like): low pass row filter bank. If none, will assume the
            same as tube filter
        h1_row (array-like): high pass row filter bank. If none, will assume the
            same as tube filter
        device: which device to put the tensors on to

    Returns:
        (h0_tube, h1_tube, h0_col, h1_col, h0_row, h1_row)
    """
    h0_tube, h1_tube = prep_filt_afb1d(h0_tube, h1_tube, device)
    if h0_col is None or h1_col is None:
        h0_col, h1_col = h0_tube, h1_tube
    else:
        h0_col, h1_col = prep_filt_afb1d(h0_col, h1_col, device)
    if h0_row is None or h1_row is None:
        h0_row, h1_row = h0_tube, h1_tube
    else:
        h0_row, h1_row = prep_filt_afb1d(h0_row, h1_row, device)

    h0_tube = h0_tube.reshape((1, 1, -1, 1, 1))
    h1_tube = h1_tube.reshape((1, 1, -1, 1, 1))
    h0_col = h0_col.reshape((1, 1, 1, -1, 1))
    h1_col = h1_col.reshape((1, 1, 1, -1, 1))
    h0_row = h0_row.reshape((1, 1, 1, 1, -1))
    h1_row = h1_row.reshape((1, 1, 1, 1, -1))
    return h0_tube, h1_tube, h0_col, h1_col, h0_row, h1_row


class AFB3D(Function):
    """ Does a single level 3d wavelet decomposition of an input. Does separate
    tube, row and column filtering by three calls to :py:func:`pytorch_wavelets.dwt.lowlevel_3d.afb1d`

    Needs to have the tensors in the right form. Because this function defines
    its own backward pass, saves on memory by not having to save the input
    tensors.

    Inputs:
        x (torch.Tensor): Input to decompose
        h0_tube: tube lowpass
        h1_tube: tube highpass
        h0_col: col lowpass
        h1_col: col highpass
        h0_row: row lowpass
        h1_row: row highpass
        mode (int): use mode_to_int to get the int code here

    We encode the mode as an integer rather than a string as gradcheck causes an
    error when a string is provided.

    Returns:
        low: Tensor of shape (N, C, D, H, W)
        highs: Tensor of shape (N, C, 7, D, H, W)
    """
    @staticmethod
    def forward(ctx, x, h0_tube, h1_tube, h0_col, h1_col, h0_row, h1_row, mode):
        ctx.save_for_backward(h0_tube, h1_tube, h0_col, h1_col, h0_row, h1_row)
        ctx.shape = x.shape[-3:]
        mode = int_to_mode(mode)
        ctx.mode = mode
        lohi_x = afb1d(x, h0_row, h1_row, mode=mode, dim=4)
        lohi_y = afb1d(lohi_x, h0_col, h1_col, mode=mode, dim=3)
        lohi_z = afb1d(lohi_y, h0_tube, h1_tube, mode=mode, dim=2)
        s = lohi_z.shape
        new_shape = (s[0], -1, 8, *s[2:])
        lohi_z = lohi_z.reshape(new_shape)
        low = lohi_z[:, :, 0].contiguous()
        highs = lohi_z[:, :, 1:].contiguous()
        return low, highs

    @staticmethod
    def backward(ctx, low, highs):
        dx = None
        if ctx.needs_input_grad[0]:
            mode = ctx.mode
            h0_tube, h1_tube, h0_col, h1_col, h0_row, h1_row = ctx.saved_tensors
            hll, lhl, hhl, llh, hlh, lhh, hhh = torch.unbind(highs, dim=2)
            ll = sfb1d(low, hll, h0_tube, h1_tube, mode=mode, dim=2)
            lh = sfb1d(llh, hlh, h0_tube, h1_tube, mode=mode, dim=2)
            hl = sfb1d(lhl, hhl, h0_tube, h1_tube, mode=mode, dim=2)
            hh = sfb1d(lhh, hhh, h0_tube, h1_tube, mode=mode, dim=2)
            l = sfb1d(ll, hl, h0_col, h1_col, mode=mode, dim=3)
            h = sfb1d(lh, hh, h0_col, h1_col, mode=mode, dim=3)
            dx = sfb1d(l, h, h0_row, h1_row, mode=mode, dim=4)
            if dx.shape[-3] > ctx.shape[-3]:
                dx = dx[:, :, :ctx.shape[-3], :, :]
            if dx.shape[-2] > ctx.shape[-2]:
                dx = dx[:, :, :, ctx.shape[-2], :]
            if dx.shape[-1] > ctx.shape[-1]:
                dx = dx[..., :ctx.shape[-1]]
        return dx, None, None, None, None, None, None, None


def prep_filt_sfb3d(g0_tube, g1_tube, g0_col=None, g1_col=None, g0_row=None, g1_row=None, device=None):
    """
    Prepares the filters to be of the right form for the sfb3d function.  In
    particular, makes the tensors the right shape. It does not mirror image them
    as as sfb3d uses conv3d_transpose which acts like normal convolution.

    Inputs:
        g0_tube (array-like): low pass tube filter bank
        g1_tube (array-like): high pass tube filter bank
        g0_col (array-like): low pass column filter bank. If none, will assume the
            same as tube filter
        g1_col (array-like): high pass column filter bank. If none, will assume the
            same as tube filter
        g0_row (array-like): low pass row filter bank. If none, will assume the
            same as tube filter
        g1_row (array-like): high pass row filter bank. If none, will assume the
            same as tube filter
        device: which device to put the tensors on to

    Returns:
        (g0_tube, g1_tube, g0_col, g1_col, g0_row, g1_row)
    """
    g0_tube, g1_tube = prep_filt_sfb1d(g0_tube, g1_tube, device)
    if g0_col is None or g1_col is None:
        g0_col, g1_col = g0_tube, g1_tube
    else:
        g0_col, g1_col = prep_filt_sfb1d(g0_col, g1_col, device)
    if g0_row is None or g1_row is None:
        g0_row, g1_row = g0_tube, g1_tube
    else:
        g0_row, g1_row = prep_filt_sfb1d(g0_row, g1_row, device)

    g0_tube = g0_tube.reshape((1, 1, -1, 1, 1))
    g1_tube = g1_tube.reshape((1, 1, -1, 1, 1))
    g0_col = g0_col.reshape((1, 1, 1, -1, 1))
    g1_col = g1_col.reshape((1, 1, 1, -1, 1))
    g0_row = g0_row.reshape((1, 1, 1, 1, -1))
    g1_row = g1_row.reshape((1, 1, 1, 1, -1))

    return g0_tube, g1_tube, g0_col, g1_col, g0_row, g1_row


class SFB3D(Function):
    """ Does a single level 3d  wavelet reconstruction of coefficients. Does separate
    tube, row and column filtering by three calls to
    :py:func:`pytorch_wavelets.dwt.lowlevel_3d.sfb1d`

    Needs to have the tensors in the right form. Because this function defines
    its own backward pass, saves on memory by not having to save the input
    tensors.

    Inputs:
        ll (torch.Tensor): approximation coefficients (N, C, D, H, W)
        highs (torch.Tensor): detail coefficients (N, C, 7, D, H, W)
        g0_tube: tube lowpass
        g1_tube: tube highpass
        g0_col: col lowpass
        g1_col: col highpass
        g0_row: row lowpass
        g1_row: row highpass
        mode (int): use mode_to_int to get the int code here

    We encode the mode as an integer rather than a string as gradcheck causes an
    error when a string is provided.

    Returns:
        y: Tensor of shape (N, C, D, H, W)
    """
    @staticmethod
    def forward(ctx, lll, highs, g0_tube, g1_tube, g0_col, g1_col, g0_row, g1_row, mode):
        ctx.save_for_backward(g0_tube, g1_tube, g0_col, g1_col, g0_row, g1_row)
        mode = int_to_mode(mode)
        ctx.mode = mode
        hll, lhl, hhl, llh, hlh, lhh, hhh = torch.unbind(highs, dim=2)
        ll = sfb1d(lll, hll, g0_tube, g1_tube, mode=mode, dim=2)
        lh = sfb1d(llh, hlh, g0_tube, g1_tube, mode=mode, dim=2)
        hl = sfb1d(lhl, hhl, g0_tube, g1_tube, mode=mode, dim=2)
        hh = sfb1d(lhh, hhh, g0_tube, g1_tube, mode=mode, dim=2)
        l = sfb1d(ll, hl, g0_col, g1_col, mode=mode, dim=3)
        h = sfb1d(lh, hh, g0_col, g1_col, mode=mode, dim=3)
        y = sfb1d(l, h, g0_row, g1_row, mode=mode, dim=4)
        return y

    @staticmethod
    def backward(ctx, dy):
        dlow, dhigh = None, None
        if ctx.needs_input_grad[0]:
            mode = ctx.mode
            g0_tube, g1_tube, g0_col, g1_col, g0_row, g1_row = ctx.saved_tensors
            dx = afb1d(dy, g0_row, g1_row, mode=mode, dim=4)
            dx = afb1d(dx, g0_col, g1_col, mode=mode, dim=3)
            dx = afb1d(dx, g0_tube, g1_tube, mode=mode, dim=2)
            s = dx.shape
            dx = dx.reshape(s[0], -1, 8, *s[2:])
            dlow = dx[:, :, 0].contiguous()
            dhigh = dx[:, :, 1:].contiguous()
        return dlow, dhigh, None, None, None, None, None, None, None
