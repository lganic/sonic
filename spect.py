import math
import librosa
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from typing import Union, Tuple


"""spect.py

Reading, and writing to audio spectrograms

Author : Logan R Boehm
"""

#for math info see : https://dspguru.com/files/Sum_of_Two_Sinusoids.pdf
def add_2_sine_waves(amplitude_1: Union[float, np.ndarray], phase_1: Union[float, np.ndarray], amplitude_2: Union[float, np.ndarray], phase_2: Union[float, np.ndarray]) -> Union[Tuple[float, float], Tuple[np.ndarray, np.ndarray]]:
    """
    Add 2 sine waves of equal frequency, and return the amplitude and phase of the new sine wave
    """
    phase_offset = phase_1 - phase_2
    amp_out = np.sqrt(math.pow(amplitude_1, 2) + math.pow(amplitude_2, 2) + 2 * amplitude_1 * amplitude_2 * np.cos(phase_offset))
    phase_out = np.arctan(amplitude_1 * np.sin(phase_offset) / (amplitude_1 * np.cos(phase_offset) + amplitude_2))
    return (amp_out, phase_out + phase_2)

class Spectrogram:
    """
    Readable, and writable spectrogram utility
    """
    def __init__(self, magnitude: np.ndarray, phase: np.ndarray, sample_rate: int = 44100, n_fft: int = 8192, hop_length: int = 16):
        self.magnitude = magnitude
        self.phase = phase
        self.strength_buffer = np.ones(self.magnitude.shape[1])
        self.sr = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length

    @staticmethod
    def from_waveform(audio_input: Union[str, np.ndarray], sample_rate: int = 44100, n_fft: int = 8192, hop_length: int = 16):
        """
        Load audio from a file or NumPy array.

        audio_input can be a string filename, or a numpy array of the waveform
        if the input is a numpy array, the sample rate can be specified with the
        sample_rate parameter. Otherwise it is determined automatically
        """

        if isinstance(audio_input, str):  # Assume it's a filepath
            y, sr = librosa.load(audio_input, sr=None)  # Load audio file
            print(np.max(y), np.min(y), np.mean(y))
        elif isinstance(audio_input, np.ndarray):  # Directly use the NumPy array
            y = audio_input
            sr = sample_rate 
        else:
            raise ValueError("Audio input must be a filepath or a NumPy array")

        stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
        
        return Spectrogram(np.abs(stft), np.angle(stft), sample_rate = sr, n_fft = n_fft, hop_length = hop_length)

    @staticmethod
    def empty(sample_rate = 44100, n_fft = 8192, hop_length = 16):
        """
        Create a new Spectrogram that is completely empty
        """
        n_y_index = (n_fft // 2) + 1
        magnitude = np.empty((n_y_index, 0))
        phase = np.empty((n_y_index, 0))
        return Spectrogram(magnitude, phase, sample_rate = sample_rate, n_fft = n_fft, hop_length = hop_length)
    
    def frequencies(self):
        """
        Generate the list of frequencies for each bin in an FFT/spectrogram.
        """
        f_res = self.sr / self.n_fft
        n_bins = (self.n_fft // 2) + 1
        frequencies = list(np.arrange(0, n_bins) * f_res)
        return frequencies
    
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
    
    def get_x_index (self, time: float) -> float:
        """
        Given a time in seconds, calculate the corresponding index in the array
        """

        return time * self.sr / self.hop_length
    
    def time_resolution(self):
        """
        Return the time delta between x values of the spectrogram
        """
        
        return self.hop_length / self.sr
    
    def ensure_length(self, index_length):
        """
        Ensure that the matrix has enough room to store the index specified
        Pad the internal matrices with zeros if it does not
        """

        ln = self.magnitude.shape[1]
        n_y_index = (self.n_fft // 2) + 1

        index_length = math.ceil(index_length)

        n_to_add = index_length - ln + 1

        if n_to_add >= 1:
            pad_out = np.zeros((n_y_index, n_to_add))
            self.magnitude = np.concatenate((self.magnitude, pad_out), axis = 1)
            self.phase = np.concatenate((self.phase, pad_out), axis = 1)
            self.strength_buffer = np.concatenate((self.strength_buffer, np.zeros(n_to_add)), axis = 1)
    
    def add_slice(self, slice_mag: np.ndarray, slice_phase: np.ndarray, index: float):
        """
        Add a slice of a spectrogram to this spectrogram,

        the index can be a floating point number, and it will be coerced to the correct position
        """

        if index < 0:
            raise IndexError("Cannot add to spectrogram, index is below zero")
        
        index_1 = int(index)
        index_2 = index_1 + 1
        index_float = index - index_1

        # Ensure that the spectrogram has enough space for the new data
        self.ensure_length(index_2)

        # Load existing data from memory
        working_mag_1 = np.sqrt(self.magnitude[:, index_1])
        working_mag_2 = np.sqrt(self.magnitude[:, index_2]) # sqrt because power spectrogram
        working_pha_1 = self.phase[:, index_1]
        working_pha_2 = self.phase[:, index_2]
        working_str_1 = self.strength_buffer[index_1]
        working_str_2 = self.strength_buffer[index_2]

        # Recreate the waveforms using the existing data
        working_mag_1 /= working_str_1
        working_mag_2 /= working_str_2

        # Add the new waveforms to the old waveforms
        output_mag_1, output_phase_1 = add_2_sine_waves((1 - index_float) * slice_mag, slice_phase, working_mag_1, working_pha_1)
        output_mag_2, output_phase_2 = add_2_sine_waves(index_float * slice_mag, slice_phase, working_mag_2, working_pha_2)

        # Update the strength buffer
        self.strength_buffer[index_1] += (1 - index_float)
        self.strength_buffer[index_2] += index_float

        # Save the new waveforms
        self.magnitude[:, index_1] = np.pow(output_mag_1 * self.strength_buffer[index_1], 2)
        self.magnitude[:, index_2] = np.pow(output_mag_2 * self.strength_buffer[index_2], 2)
        self.phase[:, index_1] = output_phase_1
        self.phase[:, index_2] = output_phase_2

    
    def duration(self):
        """
        Return the duration of the spectrogram in seconds
        """
        # Calculate the time duration per hop
        time_per_hop = self.hop_length / self.sr
    
        # Calculate the total length of the audio track
        audio_length = self.magnitude.shape[1] * time_per_hop

        return audio_length




        




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
