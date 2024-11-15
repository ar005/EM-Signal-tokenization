import numpy as np
from scipy.io import wavfile

sample_rate, audio_data = wavfile.read('noaa.wav')

#check channel count if more then 2 spplit 
if len(audio_data.shape) == 1:  # Only I (real part)
    I = audio_data
    Q = np.zeros_like(I)  # Set Q to zero if we don't have Q information
elif len(audio_data.shape) == 2 and audio_data.shape[1] == 2:  # Stereo audio: Use left and right channels for I and Q
    I = audio_data[:, 0]  # Left channel as I
    Q = audio_data[:, 1]  # Right channel as Q
else:
    raise ValueError("Unsupported WAV format. It must be mono or stereo.")


IQ_data = I + 1j * Q 

IQ_data.tofile('output_data.iq')

print(f"Conversion complete. IQ data saved to 'output_data.iq' with sample rate {sample_rate} Hz.")
