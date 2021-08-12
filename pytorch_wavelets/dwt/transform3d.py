"""
Extended by Manikandasriram S.R.
"""
import torch.nn as nn
import pywt
import pytorch_wavelets.dwt.lowlevel_3d as lowlevel
import torch


class DWTForward3D(nn.Module):
    """ Performs a 3d DWT forward decomposition of a tensor

    Args:
        J (int): Number of levels of decomposition
        wave (str or pywt.Wavelet or tuple(ndarray)): Which wavelet to use.
            Can be:
            1) a string to pass to pywt.Wavelet constructor
            2) a pywt.Wavelet class
            3) a tuple of numpy arrays, either (h0, h1) or (h0_tube, h1_tube, h0_col, h1_col, h0_row, h1_row)
        mode (str): 'zero', 'symmetric', 'reflect' or 'periodization'. The
            padding scheme
        """
    def __init__(self, J=1, wave='db1', mode='zero'):
        super(DWTForward3D, self).__init__()
        if isinstance(wave, str):
            wave = pywt.Wavelet(wave)
        if not wave.name == 'db1' and not wave.name == 'haar':
            raise NotImplementedError()
        if isinstance(wave, pywt.Wavelet):
            h0_tube, h1_tube = wave.dec_lo, wave.dec_hi
            h0_col, h1_col = h0_tube, h1_tube
            h0_row, h1_row = h0_col, h1_col
        else:
            if len(wave) == 2:
                h0_tube, h1_tube = wave[0], wave[1]
                h0_col, h1_col = h0_tube, h1_tube
                h0_row, h1_row = h0_col, h1_col
            elif len(wave) == 6:
                h0_tube, h1_tube = wave[0], wave[1]
                h0_col, h1_col = wave[2], wave[3]
                h0_row, h1_row = wave[4], wave[5]
            else:
                raise ValueError(f"Got {len(wave)} decoding filters but expected 2 or 6.")

        # Prepare the filters
        filts = lowlevel.prep_filt_afb3d(h0_tube, h1_tube, h0_col, h1_col, h0_row, h1_row)
        self.register_buffer('h0_tube', filts[0])
        self.register_buffer('h1_tube', filts[1])
        self.register_buffer('h0_col', filts[2])
        self.register_buffer('h1_col', filts[3])
        self.register_buffer('h0_row', filts[4])
        self.register_buffer('h1_row', filts[5])
        self.J = J
        self.mode = mode

    def forward(self, x):
        """ Forward pass of the DWT.

        Args:
            x (tensor): Input of shape :math: `(N, C_{in}, D_{in}, H_{in}, W_{in})`

        Returns:
            (yl, yh)
                tuple of lowpass (yl) and bandpass (yh) coefficients.
                yh is a list of length J with the first entry
                being the finest scale coefficients. yl has shape
                :math:`(N, C_{in}, D_{in}', H_{in}', W_{in}')` and yh has shape
                :math:`list(N, C_{in}, 7, D_{in}'', H_{in}'', W_{in}'')`. The new
                dimension in yh iterates over the LLH, LHL, LHH, HLL, HLH, HHL and HHH coefficients.

        Note:
            :math:`(D_{in}', H_{in}', W_{in}'), (D_{in}'', H_{in}'', W_{in}'')` denote the correctly
            downsampled shapes of the DWT pyramid.
        """
        yh = []
        ll = x
        mode = lowlevel.mode_to_int(self.mode)

        # Do a multilevel transform
        for j in range(self.J):
            # Do 1 level of the transform
            ll, high = lowlevel.AFB3D.apply(ll, self.h0_tube, self.h1_tube, self.h0_col, self.h1_col, self.h0_row,
                                            self.h1_row, mode)
            yh.append(high)

        return ll, yh


class DWTInverse3D(nn.Module):
    """ Performs a 3d DWT Inverse reconstruction of a tensor

    Args:
        wave (str or pywt.Wavelet or tuple(ndarray)): Which wavelet to use.
            Can be:
            1) a string to pass to pywt.Wavelet constructor
            2) a pywt.Wavelet class
            3) a tuple of numpy arrays, either (h0, h1) or (h0_tube, h1_tube, h0_col, h1_col, h0_row, h1_row)
        mode (str): 'zero', 'symmetric', 'reflect' or 'periodization'. The
            padding scheme
    """
    def __init__(self, wave='db1', mode='zero'):
        super(DWTInverse3D, self).__init__()
        if isinstance(wave, str):
            wave = pywt.Wavelet(wave)
        if not wave.name == 'db1' and not wave.name == 'haar':
            raise NotImplementedError()
        if isinstance(wave, pywt.Wavelet):
            g0_tube, g1_tube = wave.rec_lo, wave.rec_hi
            g0_col, g1_col = g0_tube, g1_tube
            g0_row, g1_row = g0_col, g1_col
        else:
            if len(wave) == 2:
                g0_tube, g1_tube = wave[0], wave[1]
                g0_col, g1_col = g0_tube, g1_tube
                g0_row, g1_row = g0_col, g1_col
            elif len(wave) == 6:
                g0_tube, g1_tube = wave[0], wave[1]
                g0_col, g1_col = wave[2], wave[3]
                g0_row, g1_row = wave[4], wave[5]

        # Prepare the filters
        filts = lowlevel.prep_filt_sfb3d(g0_tube, g1_tube, g0_col, g1_col, g0_row, g1_row)
        self.register_buffer('g0_tube', filts[0])
        self.register_buffer('g1_tube', filts[1])
        self.register_buffer('g0_col', filts[2])
        self.register_buffer('g1_col', filts[3])
        self.register_buffer('g0_row', filts[4])
        self.register_buffer('g1_row', filts[5])
        self.mode = mode

    def forward(self, coeffs):
        """
        Args:
            coeffs (yl, yh): tuple of lowpass and bandpass coefficients, where:
              yl is a lowpass tensor of shape :math:`(N, C_{in}, D_{in}', H_{in}',
              W_{in}')` and yh is a list of bandpass tensors of shape
              :math:`list(N, C_{in}, 7, D_{in}'', H_{in}'', W_{in}'')`. i.e. should match
              the format returned by DWTForward3D

        Returns:
            Reconstructed input of shape :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})`

        Note:
            :math:`(D_{in}', H_{in}', W_{in}'), (D_{in}'', H_{in}'', W_{in}'')` denote the correctly
            downsampled shapes of the DWT pyramid.

        Note:
            Can have None for any of the highpass scales and will treat the
            values as zeros (not in an efficient way though).
        """
        yl, yh = coeffs
        ll = yl
        mode = lowlevel.mode_to_int(self.mode)

        # Do a multilevel inverse transform
        for h in yh[::-1]:
            if h is None:
                h = torch.zeros(ll.shape[0], ll.shape[1], 7, ll.shape[-3], ll.shape[-2],
                                ll.shape[-1], device=ll.device)

            # 'Unpad' added dimensions
            if ll.shape[-3] > h.shape[-3]:
                ll = ll[..., :-1, :, :]
            if ll.shape[-2] > h.shape[-2]:
                ll = ll[..., :-1, :]
            if ll.shape[-1] > h.shape[-1]:
                ll = ll[..., :-1]
            ll = lowlevel.SFB3D.apply(ll, h, self.g0_tube, self.g1_tube, self.g0_col, self.g1_col, self.g0_row,
                                      self.g1_row, mode)
        return ll
