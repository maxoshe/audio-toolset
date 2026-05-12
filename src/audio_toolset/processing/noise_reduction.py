import numpy as np
from scipy.fftpack import fft, ifft

from audio_toolset.audio_data import AudioData
from audio_toolset.util import convert_db_to_factor


def _apply_spectral_gating(
    audio_data: AudioData, noise_threshold: float, attenuation: float
) -> AudioData:
    new_data = audio_data.get_copy()
    signal_fft = fft(new_data.data)
    sample_size = new_data.get_number_of_samples()
    signal_fft[abs(signal_fft) * (2 / sample_size) < noise_threshold] *= attenuation
    new_data.data = np.real(ifft(signal_fft))
    return new_data


def _apply_wiener_filter(
    audio_data: AudioData, noise_threshold: float, attenuation: float
) -> AudioData:
    new_data = audio_data.get_copy()
    signal_fft = fft(new_data.data)
    sample_size = new_data.get_number_of_samples()
    normalized_magnitude = np.maximum(abs(signal_fft) * (2 / sample_size), 1e-10)
    gain = np.maximum(attenuation, 1 - (noise_threshold / normalized_magnitude) ** 2)
    signal_fft *= gain
    new_data.data = np.real(ifft(signal_fft))
    return new_data


def apply_noise_reduction(
    audio_data: AudioData,
    noise_threshold_db: float = -50,
    attenuation_db: float = -1,
    smooth: bool = True,
) -> AudioData:
    noise_threshold = convert_db_to_factor(noise_threshold_db)
    attenuation = convert_db_to_factor(attenuation_db)

    if smooth:
        return _apply_wiener_filter(audio_data, noise_threshold, attenuation)
    return _apply_spectral_gating(audio_data, noise_threshold, attenuation)
