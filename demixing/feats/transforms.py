import math
import torch
import torch.nn as nn
from . import functional as F
from typing import Callable, Optional


class MFCC(nn.Module):

    __constants__ = ['sample_rate', 'n_mfcc', 'dct_type', 'top_db', 'log_mels']

    def __init__(
            self,
            sample_rate: int = 16000,
            n_mfcc: int = 40,
            dct_type: int = 2,
            norm: str = 'ortho',
            log_mels: bool = False,
            melkwargs: Optional[dict] = None
        ):
        super(MFCC, self).__init__()
        supported_dct_types = [2]
        if dct_type not in supported_dct_types:
            raise ValueError('DCT type not supported: {}'.format(dct_type))
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.dct_type = dct_type
        self.norm = norm
        self.top_db = 80.0
        self.amplitude_to_DB = AmplitudeToDB('power', self.top_db)

        melkwargs = melkwargs or {}
        self.melspectrogram = MelSpectrogram(sample_rate=self.sample_rate, **melkwargs)

        if self.n_mfcc > self.melspectrogram.n_mels:
            raise ValueError('Cannot select more MFCC coefficients than # mel bins')
        dct_mat = F.create_dct(self.n_mfcc, self.melspectrogram.n_mels, self.norm)
        self.register_buffer('dct_mat', dct_mat)
        self.log_mels = log_mels

    def forward(self, waveform: torch.Tensor):
        """
        Parameters
        ----------
        waveform: torch.Tensor
            Tensor of audio of dimension [batch, time].
        
        Returns
        -------
        mfcc: torch.Tensor
            specgram_mel_db of size (batch, n_mfcc, time).
        """
        mel_specgram = self.melspectrogram(waveform)
        if self.log_mels:
            log_offset = 1e-6
            mel_specgram = torch.log(mel_specgram + log_offset)
        else:
            mel_specgram = self.amplitude_to_DB(mel_specgram)

        # (batch, time, n_mels) dot (n_mels, n_mfcc) -> (batch, n_nfcc, time)
        mfcc = torch.matmul(mel_specgram.transpose(-1, -2), self.dct_mat).transpose(-1, -2)
        return mfcc


class MelSpectrogram(torch.nn.Module):
    """
    Parameters
    ----------
    sample_rate (int, optional): 
        Sample rate of audio signal. (Default: ``16000``)
    n_fft (int, optional): 
        Size of FFT, creates ``n_fft // 2 + 1`` bins. (Default: ``400``)
    win_length (int or None, optional): 
        Window size. (Default: ``n_fft``)
    hop_length (int or None, optional): 
        Length of hop between STFT windows. (Default: ``win_length // 2``)
    f_min (float, optional): 
        Minimum frequency. (Default: ``0.``)
    f_max (float or None, optional): 
        Maximum frequency. (Default: ``None``)
    pad (int, optional): 
        Two sided padding of signal. (Default: ``0``)
    n_mels (int, optional): 
        Number of mel filterbanks. (Default: ``128``)
    window_fn (Callable[..., Tensor], optional): 
        A function to create a window tensor
        that is applied/multiplied to each frame/window. (Default: ``torch.hann_window``)
    power (float, optional): 
        Exponent for the magnitude spectrogram,
        (must be > 0) e.g., 1 for energy, 2 for power, etc. (Default: ``2``)
    normalized (bool, optional): 
        Whether to normalize by magnitude after stft. (Default: ``False``)
    wkwargs (Dict[..., ...] or None, optional): 
        Arguments for window function. (Default: ``None``)
    center (bool, optional): 
        whether to pad :attr:`waveform` on both sides so
        that the :math:`t`-th frame is centered at time :math:`t \times \text{hop\_length}`.
        (Default: ``True``)
    pad_mode (string, optional): 
        controls the padding method used when
        :attr:`center` is ``True``. (Default: ``"reflect"``)
    onesided (bool, optional): 
        controls whether to return half of results to
        avoid redundancy. (Default: ``True``)
    norm (str or None, optional): 
        If 'slaney', divide the triangular mel weights by the width of the mel band
        (area normalization). (Default: ``None``)
    mel_scale (str, optional): 
        Scale to use: ``htk`` or ``slaney``. (Default: ``htk``)
    
    Examples
    --------
    >>> waveform, sample_rate = torchaudio.load('test.wav', normalize=True)
    >>> transform = transforms.MelSpectrogram(sample_rate)
    >>> mel_specgram = transform(waveform)  # (channel, n_mels, time)
    """

    __constants__ = ['sample_rate', 'n_fft', 'win_length', 'hop_length', 'pad', 'n_mels', 'f_min']

    def __init__(
            self,
            sample_rate: int = 16000,
            n_fft: int = 400,
            win_length: Optional[int] = None,
            hop_length: Optional[int] = None,
            f_min: float = 0.,
            f_max: Optional[float] = None,
            pad: int = 0,
            n_mels: int = 128,
            window_fn: Callable[..., torch.Tensor] = torch.hann_window,
            power: float = 2.,
            normalized: bool = False,
            wkwargs: Optional[dict] = None,
            center: bool = True,
            pad_mode: str = "reflect",
            onesided: bool = True,
            norm: Optional[str] = None,
            mel_scale: str = "htk"
        ):
        super(MelSpectrogram, self).__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length if win_length is not None else n_fft
        self.hop_length = hop_length if hop_length is not None else self.win_length // 2
        self.pad = pad
        self.power = power
        self.normalized = normalized
        self.n_mels = n_mels
        self.f_max = f_max
        self.f_min = f_min
        self.spectrogram = Spectrogram(n_fft=self.n_fft, win_length=self.win_length,
                                       hop_length=self.hop_length,
                                       pad=self.pad, window_fn=window_fn, power=self.power,
                                       normalized=self.normalized, wkwargs=wkwargs,
                                       center=center, pad_mode=pad_mode, onesided=onesided)
        self.mel_scale = MelScale(
            self.n_mels,
            self.sample_rate,
            self.f_min,
            self.f_max,
            self.n_fft // 2 + 1,
            norm,
            mel_scale
        )

    def forward(self, waveform):
        """
        Parameters
        ----------
        waveform: torch.Tensor
            Tensor of audio of dimension (batch, time).
        
        Returns
        -------
        mel_specgram: torch.Tensor
            Mel frequency spectrogram of size (batch, n_mels, time).
        """
        specgram = self.spectrogram(waveform)
        mel_specgram = self.mel_scale(specgram)
        return mel_specgram


class MelScale(torch.nn.Module):
    """
    Parameters
    ----------
        n_mels (int, optional): Number of mel filterbanks. (Default: ``128``)
        sample_rate (int, optional): Sample rate of audio signal. (Default: ``16000``)
        f_min (float, optional): Minimum frequency. (Default: ``0.``)
        f_max (float or None, optional): Maximum frequency. (Default: ``sample_rate // 2``)
        n_stft (int, optional): Number of bins in STFT. See ``n_fft`` in :class:`Spectrogram`. (Default: ``201``)
        norm (str or None, optional): If 'slaney', divide the triangular mel weights by the width of the mel band
            (area normalization). (Default: ``None``)
        mel_scale (str, optional): Scale to use: ``htk`` or ``slaney``. (Default: ``htk``)
    """
    __constants__ = ['n_mels', 'sample_rate', 'f_min', 'f_max']

    def __init__(
            self,
            n_mels: int = 128,
            sample_rate: int = 16000,
            f_min: float = 0.,
            f_max: Optional[float] = None,
            n_stft: int = 201,
            norm: Optional[str] = None,
            mel_scale: str = "htk"
        ):
        super(MelScale, self).__init__()
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.f_max = f_max if f_max is not None else float(sample_rate // 2)
        self.f_min = f_min
        self.norm = norm
        self.mel_scale = mel_scale

        assert f_min <= self.f_max, 'Require f_min: {} < f_max: {}'.format(f_min, self.f_max)
        fb = F.melscale_fbanks(
            n_stft, 
            self.f_min, 
            self.f_max, 
            self.n_mels, 
            self.sample_rate, 
            self.norm,
            self.mel_scale
        )
        self.register_buffer('fb', fb)

    def forward(self, specgram: torch.Tensor):
        """
        Parameters
        ----------
        specgram (Tensor): 
            A spectrogram STFT of dimension (..., freq, time).
        
        Returns
        -------
            Tensor: Mel frequency spectrogram of size (batch, ``n_mels``, time).
        """

        # (batch, time, freq) dot (freq, n_mels) -> (batch, n_mels, time)
        mel_specgram = torch.matmul(specgram.transpose(-1, -2), self.fb).transpose(-1, -2)
        return mel_specgram


class Spectrogram(torch.nn.Module):
    """
    Parameters
    ----------
        n_fft (int, optional): Size of FFT, creates ``n_fft // 2 + 1`` bins. (Default: ``400``)
        win_length (int or None, optional): Window size. (Default: ``n_fft``)
        hop_length (int or None, optional): Length of hop between STFT windows. (Default: ``win_length // 2``)
        pad (int, optional): Two sided padding of signal. (Default: ``0``)
        window_fn (Callable[..., Tensor], optional): A function to create a window tensor
            that is applied/multiplied to each frame/window. (Default: ``torch.hann_window``)
        power (float or None, optional): Exponent for the magnitude spectrogram,
            (must be > 0) e.g., 1 for energy, 2 for power, etc.
            If None, then the complex spectrum is returned instead. (Default: ``2``)
        normalized (bool, optional): Whether to normalize by magnitude after stft. (Default: ``False``)
        wkwargs (dict or None, optional): Arguments for window function. (Default: ``None``)
        center (bool, optional): whether to pad :attr:`waveform` on both sides so
            that the :math:`t`-th frame is centered at time :math:`t \times \text{hop\_length}`.
            (Default: ``True``)
        pad_mode (string, optional): controls the padding method used when
            :attr:`center` is ``True``. (Default: ``"reflect"``)
        onesided (bool, optional): controls whether to return half of results to
            avoid redundancy (Default: ``True``)
        return_complex (bool, optional):
            Indicates whether the resulting complex-valued Tensor should be represented with
            native complex dtype, such as `torch.cfloat` and `torch.cdouble`, or real dtype
            mimicking complex value with an extra dimension for real and imaginary parts.
            (See also ``torch.view_as_real``.)
            This argument is only effective when ``power=None``. It is ignored for
            cases where ``power`` is a number as in those cases, the returned tensor is
            power spectrogram, which is a real-valued tensor.
    
    Examples
    --------
    >>> waveform, sample_rate = torchaudio.load('test.wav', normalize=True)
    >>> transform = torchaudio.transforms.Spectrogram(n_fft=800)
    >>> spectrogram = transform(waveform)
    """
    __constants__ = ['n_fft', 'win_length', 'hop_length', 'pad', 'power', 'normalized']

    def __init__(self,
                 n_fft: int = 400,
                 win_length: Optional[int] = None,
                 hop_length: Optional[int] = None,
                 pad: int = 0,
                 window_fn: Callable[..., torch.Tensor] = torch.hann_window,
                 power: Optional[float] = 2.,
                 normalized: bool = False,
                 wkwargs: Optional[dict] = None,
                 center: bool = True,
                 pad_mode: str = "reflect",
                 onesided: bool = True,
                 return_complex: bool = True):
        super(Spectrogram, self).__init__()
        self.n_fft = n_fft
        # number of FFT bins. the returned STFT result will have n_fft // 2 + 1
        # number of frequencies due to onesided=True in torch.stft
        self.win_length = win_length if win_length is not None else n_fft
        self.hop_length = hop_length if hop_length is not None else self.win_length // 2
        window = window_fn(self.win_length) if wkwargs is None else window_fn(self.win_length, **wkwargs)
        self.register_buffer('window', window)
        self.pad = pad
        self.power = power
        self.normalized = normalized
        self.center = center
        self.pad_mode = pad_mode
        self.onesided = onesided
        self.return_complex = return_complex

    def forward(self, waveform: torch.Tensor):
        """
        Parameters
        ----------
        waveform (Tensor): 
            Tensor of audio of dimension (..., time).
        
        Returns
        -------
        Tensor: Dimension (..., freq, time), where freq is
        ``n_fft // 2 + 1`` where ``n_fft`` is the number of
        Fourier bins, and time is the number of window hops (n_frame).
        """
        return F.spectrogram(
            waveform,
            self.pad,
            self.window,
            self.n_fft,
            self.hop_length,
            self.win_length,
            self.power,
            self.normalized,
            self.center,
            self.pad_mode,
            self.onesided,
            self.return_complex,
        )



class AmplitudeToDB(nn.Module):
    
    __constants__ = ['multiplier', 'amin', 'ref_value', 'db_multiplier']

    def __init__(self, stype: str = 'power', top_db: Optional[float] = None):
        super(AmplitudeToDB, self).__init__()
        self.stype = stype
        if top_db is not None and top_db < 0:
            raise ValueError('top_db must be positive value')
        self.top_db = top_db
        self.multiplier = 10.0 if stype == 'power' else 20.0
        self.amin = 1e-10
        self.ref_value = 1.0
        self.db_multiplier = math.log10(max(self.amin, self.ref_value))

    def forward(self, x):
        """
        Parameters
        ----------
        x: torch.Tensor
            Input tensor before being converted to decibel scale.
        
        Returns
        -------
        output: torch.Tensor
            Output tensor in decibel scale.
        """
        output = F.amplitude_to_DB(x, self.multiplier, self.amin, self.db_multiplier, self.top_db)
        return output