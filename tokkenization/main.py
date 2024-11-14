# for tokenizing the signals

import numpy as np
from scipy.signal import detrend
from scipy.io import wavfile

def tokenize_em_signal(signal, num_tokens=500):
    """
    Tokenizes the EM signal using the Discrete Fourier Transform (DFT) 
    and converts it into frequency tokens.

    Args:
        signal (numpy array): Preprocessed EM signal.
        num_tokens (int): Number of frequency tokens to generate.

    Returns:
        numpy array: Tokenized signal in the frequency domain.
    """
    # Step 1: Apply the Discrete Fourier Transform (DFT) to convert the signal to the frequency domain
    frequency_components = np.fft.fft(signal)

    # Step 2: Compute the magnitude of each frequency component
    magnitudes = np.abs(frequency_components)

    # Step 3: Normalize the frequency components to create tokens
    tokens = magnitudes[:num_tokens]  # Take the first 'num_tokens' components for simplicity

    return tokens

def preprocess_em_signal(em_signal):
    """
    Preprocesses the EM signal by detrending and normalizing it.

    Args:
        em_signal (numpy array): Raw EM signal data (1D time-series).

    Returns:
        numpy array: Preprocessed EM signal.
    """
    # Step 1: Remove linear trend from the signal (optional but useful for some signals)
    detrended_signal = detrend(em_signal)

    # Step 2: Normalize the signal to have zero mean and unit variance
    normalized_signal = (detrended_signal - np.mean(detrended_signal)) / np.std(detrended_signal)

    return normalized_signal

def read_wav_file(file_path):
    """
    Reads a .wav file and returns the audio data as a numpy array.

    Args:
        file_path (str): Path to the .wav file.

    Returns:
        numpy array: The audio signal data.
    """
    # Load the .wav file
    sample_rate, signal_data = wavfile.read(file_path)

    # If the signal has more than one channel (stereo), take only the first channel
    if len(signal_data.shape) > 1:
        signal_data = signal_data[:, 0]  # Use the first channel for simplicity

    return signal_data

# Example usage
file_path = 'baseband_137040555Hz_10-50-39_27-06-2024.wav'  

raw_em_signal = read_wav_file(file_path)
preprocessed_signal = preprocess_em_signal(raw_em_signal)
#print(preprocessed_signal.size())
print(preprocessed_signal.shape)
print(preprocessed_signal)


tokens = tokenize_em_signal(preprocessed_signal)
print(tokens)
