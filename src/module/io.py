from scipy.io import wavfile
import os
import numpy as np


class io:
    def __init__(self) -> None:
        pass

    def save_wav(self, filepath: str, sample_rate: int, audio_data_64bf: np.ndarray):
        """
        Saves a NumPy array as a WAV file.

        Parameters
        ----------
        filepath : str
            The path where the WAV file will be saved. Must have a ".wav" extension.
        sample_rate : int
            The sample rate of the audio data.
        audio_data_64bf : np.ndarray
            The audio data to be saved, in 64-bit float format.

        Raises
        ------
        AssertionError
            If the file extension is not ".wav".
        """
        assert os.path.splitext(filepath)[1] == ".wav", f"file is not wav: {filepath}"
        wavfile.write(filepath, sample_rate, audio_data_64bf)

    def load_wav_as_mono(filepath: str):
        """
        Loads a WAV file and returns the audio data as mono.

        Parameters
        ----------
        filepath : str
            The path to the WAV file to be loaded.

        Returns
        -------
        tuple
            A tuple containing the sample rate (int) and the audio data (np.ndarray) as a mono channel.

        Raises
        ------
        AssertionError
            If the file does not exist or if the file extension is not ".wav".
        """
        assert os.path.exists(filepath), f"file not found: {filepath}"
        assert os.path.splitext(filepath)[1] == ".wav", f"file is not wav: {filepath}"

        sample_rate, audio = wavfile.read(filepath)
        estimated_int_bit_depth = 0
        if audio.dtype == np.int16:
            estimated_int_bit_depth = 16
        elif audio.dtype == np.int32:
            if np.max(audio) > 2**15:
                estimated_int_bit_depth = 32
            else:
                estimated_int_bit_depth = 24

        audio = audio.astype(np.longdouble)
        if estimated_int_bit_depth != 0:
            audio /= 2 ** (estimated_int_bit_depth - 1)

        # if odd length, add 0 to last
        if audio.shape[0] % 2 == 1:
            audio = np.concatenate(
                [audio, np.zeros((1, audio.shape[1]), dtype=np.longdouble)],
                dtype=np.longdouble,
            )

        # print(audio.shape)
        audio_data_per_channel = audio if len(audio.shape) == 1 else audio[:, 0]
        return sample_rate, audio_data_per_channel
