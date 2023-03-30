import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import pywt
from scipy import signal
from scipy.signal import correlate, spectrogram


# Set up parameters for STFT
SAMPLING_RATE = 1280
num_frames=64
nfft = 256

# Set up path for your data
data_path = 'drone motion state 3s framed dataset'

# Set up dictionary to store spectrograms and class labels
spectrograms_dict = {'Azimuthal':[],'Hovering': [], 'Range': []}

# Loop through each class folder
for class_label in ['Azimuthal','Hovering', 'Range']:
    
    # Set up path for class folder
    class_path = os.path.join(data_path, class_label)
    
    # Loop through each csv file in the class folder
    for csv_file in os.listdir(class_path):
        
        # Load csv file
        data = pd.read_csv(os.path.join(class_path, csv_file), header=None)

        # Split the data into I and Q signals
        I = data.iloc[:, 0]
        Q = data.iloc[:, 1]


        # Combine the I and Q signals into a complex signal
        complex_signal = I + 1j * Q

        # Calculate the window length and hop length
        window_length = int(len(I) / num_frames)
        hop_length = int(window_length / 2)

        # Calculate the STFT of the complex signal
        f, t, Zxx = signal.stft(complex_signal, fs=SAMPLING_RATE, window='hann', nperseg=window_length, noverlap=window_length-hop_length, nfft=nfft, return_onesided=False)
       

        # Calculate the magnitude squared of the STFT coefficients
        spectrogram = np.abs(Zxx)**2
        
        # Store spectrogram and class label in dictionary
        spectrograms_dict[class_label].append(spectrogram)

    import numpy as np

# Store spectrograms dictionary as .npz file
np.savez('spectrograms_dict.npz', **spectrograms_dict)


