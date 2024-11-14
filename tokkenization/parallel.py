import numpy as np
from scipy.signal import detrend
import soundfile as sf
from concurrent.futures import ProcessPoolExecutor, as_completed

def tokenize_chunk(chunk, num_tokens):
    """
    Tokenizes a chunk using the Discrete Fourier Transform (DFT) 
    and returns frequency tokens.
    
    Args:
        chunk (numpy array): A segment of the preprocessed signal.
        num_tokens (int): Number of frequency tokens to generate per chunk.

    Returns:
        numpy array: Frequency tokens for the chunk.
    """
    # Apply the Discrete Fourier Transform (DFT) to convert the chunk to the frequency domain
    frequency_components = np.fft.fft(chunk)
    magnitudes = np.abs(frequency_components)
    tokens = magnitudes[:num_tokens] if num_tokens < len(magnitudes) else magnitudes
    return tokens

def tokenize_em_signal_parallel(signal, num_tokens=500, chunk_size=1000000):
    """
    Tokenizes the EM signal in parallel by splitting it into chunks.
    
    Args:
        signal (numpy array): Preprocessed EM signal.
        num_tokens (int): Total number of frequency tokens to generate.
        chunk_size (int): Size of each chunk for parallel processing.

    Returns:
        numpy array: Aggregated tokens from all chunks.
    """
    # Split the signal into chunks
    chunks = [signal[i:i + chunk_size] for i in range(0, len(signal), chunk_size)]
    
    # Process chunks in parallel
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(tokenize_chunk, chunk, num_tokens) for chunk in chunks]
        tokens = []
        
        # Aggregate tokens as chunks finish processing
        for future in as_completed(futures):
            tokens.extend(future.result())
            
    # Take only the first num_tokens from the aggregated result to match desired length
    return np.array(tokens[:num_tokens])

def process_chunk(chunk):
    """
    Processes a chunk by detrending and normalizing.

    Args:
        chunk (numpy array): A chunk of the EM signal.

    Returns:
        numpy array: Processed chunk of the EM signal.
    """
    # Detrend and normalize the chunk
    detrended = detrend(chunk)
    return (detrended - np.mean(detrended)) / np.std(detrended)

def preprocess_em_signal(em_signal, num_chunks=4):
    """
    Preprocesses the EM signal by splitting into chunks, detrending, and normalizing in parallel.

    Args:
        em_signal (numpy array): Raw EM signal data (1D time-series).
        num_chunks (int): Number of chunks to split the signal into for parallel processing.

    Returns:
        numpy array: Preprocessed EM signal.
    """
    # Split signal into chunks
    chunks = np.array_split(em_signal, num_chunks)

    # Process chunks in parallel
    with ProcessPoolExecutor() as executor:
        processed_chunks = list(executor.map(process_chunk, chunks))

    # Combine processed chunks
    return np.concatenate(processed_chunks)

def read_wav_file(file_path):
    """
    Reads a .wav file and returns the audio data as a numpy array.

    Args:
        file_path (str): Path to the .wav file.

    Returns:
        numpy array: The audio signal data.
    """
    try:
        # Load the .wav file with soundfile
        signal_data, sample_rate = sf.read(file_path)

        # If the signal has more than one channel, take only the first channel
        if len(signal_data.shape) > 1:
            signal_data = signal_data[:, 0]  # Use the first channel for simplicity

        return signal_data
    except Exception as e:
        print(f"Error reading the WAV file: {e}")
        return None

# Example usage
#file_path = 'baseband_137529087Hz_15-09-40_18-06-2024.wav'

file_path = 'baseband_137040555Hz_10-50-39_27-06-2024.wav' 
raw_em_signal = read_wav_file(file_path)

if raw_em_signal is not None:
    # Preprocess the signal in parallel
    preprocessed_signal = preprocess_em_signal(raw_em_signal)
    print(f"Preprocessed Signal Shape: {preprocessed_signal.shape}")

    # Tokenize the signal in parallel
    tokens = tokenize_em_signal_parallel(preprocessed_signal)
    print("Tokens:", tokens)
else:
    print("Failed to read the WAV file.")
