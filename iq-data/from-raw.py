import numpy as np

def read_iq_data_from_raw(file_path, num_samples, data_type=np.int16):
 

    with open(file_path, 'rb') as f:
        # Read the raw binary data (2 bytes for I and 2 bytes for Q per sample)
        iq_raw = np.fromfile(f, dtype=data_type, count=2 * num_samples)
        #iq_raw = np.fromfile(f, dtype=np.float32, count=2 * num_samples) #for 32
    
    if iq_raw.size != 2 * num_samples:
        raise ValueError(f"File contains an unexpected number of samples. Expected {2 * num_samples}, got {iq_raw.size}.")
    
    # Split the raw data into I and Q 
    I = iq_raw[::2]  # Take every second sample starting from 0 (I values)
    Q = iq_raw[1::2]  # Take every second sample starting from 1 (Q values)

    # Combine I and Q into complex IQ data (I + jQ)
    iq_data = I + 1j * Q
    
    return iq_data

file_path = 'meteor.raw'  
num_samples = 100000  # Number of complex samples to read 
iq_data = read_iq_data_from_raw(file_path, num_samples)

print(iq_data[:10]) 
