import librosa
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from typing import Union

"""spect.py

Reading, and writing to audio spectrograms

Author : Logan R Boehm
"""

class Spectrogram:
    """
    Readable, and writable spectrogram utility
    """
    def __init__(self, audio_input: Union[str, np.ndarray], sample_rate: int = 44100, n_fft: int = 8192, hop_length: int = 16):
        """
        Load audio from a file or NumPy array.

        audio_input can be a string filename, or a numpy array of the waveform
        if the input is a numpy array, the sample rate can be specified with the
        sample_rate parameter. Otherwise it is determined automatically
        """
        
        if isinstance(audio_input, str):  # Assume it's a filepath
            y, sr = librosa.load(audio_input, sr=None)  # Load audio file
        elif isinstance(audio_input, np.ndarray):  # Directly use the NumPy array
            y = audio_input
            sr = sample_rate
        else:
            raise ValueError("Audio input must be a filepath or a NumPy array")
        
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
        self.magnitude, self.phase = np.abs(stft), np.angle(stft)
    
    def preview(self, filename: str)-> None:
        """
        Save the current spectrogram as an image
        """
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(librosa.amplitude_to_db(self.magnitude, ref=np.max),
                             sr=self.sr, hop_length=self.hop_length, y_axis='log', x_axis='time')
        plt.title('Spectrogram')
        plt.colorbar(format='%+2.0f dB')
        plt.tight_layout()
        plt.savefig(filename)

    def reconstruct_audio(self)-> np.ndarray:
        """
        Reconstruct audio from magnitude and phase spectrograms.
        """
        # Combine magnitude and phase to get the complex STFT matrix
        stft_matrix = self.magnitude * np.exp(1j * self.phase)
    
        # Perform the inverse STFT
        y_reconstructed = librosa.istft(stft_matrix, hop_length=self.hop_length)
    
        return y_reconstructed
    
    def save(self, filename: str)-> None:
        """
        Reconstruct, and save the audio as the filename
        """

        reconstruct = self.reconstruct_audio()

        sf.write(filename, reconstruct, self.sr)
    
    def get_y_index (self, frequency: float)-> float:
        """
        Given a frequency in Hz, calculate the corresponding index in the array
        """

        return frequency * self.n_fft / self.sr


if __name__ == '__main__':
    audio_input = 'kalaz.wav'
    spect = Spectrogram(audio_input)
    print(spect.magnitude.shape)
#    for x in range(spect.magnitude.shape[1]):
#        for y in range(spect.magnitude.shape[0]):
#            if y > 1000:
#                spect.magnitude[y, x] = 0
    spect.preview('prev.png')
    spect.save('recon.wav')
