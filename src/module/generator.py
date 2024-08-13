import numpy as np
import scipy.signal
import os
import module.io as io


class generator:
    def __init__(
        self,
        sample_rate: float,
        output_dir: str,
    ) -> None:
        self.sample_rate = sample_rate
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.io = io.io()

    def generate_sweep_up(
        self,
        length: int,
        start_frequency: float,
        end_frequency: float,
        log_scale: bool = True,
    ):
        """
        Generates a sine wave sweep signal and saves it as a WAV file.

        The sine wave sweep is generated with the specified start and end frequencies and duration.

        Parameters
        ----------
        length : int
            The length of the sine wave sweep signal, in samples.
        start_frequency : float
            The start frequency of the sine wave sweep, in Hertz.
        end_frequency : float
            The end frequency of the sine wave sweep, in Hertz.
        log_scale : bool, optional
            If True, the sweep is generated in a logarithmic scale. If False, it is generated in a linear scale.

        Returns
        -------
        np.ndarray
            The sine wave sweep signal as a NumPy array.
        """
        t = np.arange(length, dtype=np.longdouble) / self.sample_rate

        if log_scale:
            sweep = scipy.signal.chirp(
                t, start_frequency, t[-1], end_frequency, method="logarithmic", phi=-90
            )
        else:
            sweep = scipy.signal.chirp(
                t, start_frequency, t[-1], end_frequency, method="linear", phi=-90
            )

        self.io.save_wav(
            os.path.join(self.output_dir, "sweep.wav"),
            self.sample_rate,
            sweep.astype(np.double),
        )

        return sweep

    def generate_impulse(self, length: int):
        """
        Generates an impulse signal and saves it as a WAV file.

        The impulse is a signal with a single value of 1 at the center of the array,
        and 0 elsewhere.

        Parameters
        ----------
        length : int
            The length of the impulse signal, in samples.

        Returns
        -------
        np.ndarray
            The impulse signal as a NumPy array.
        """
        impulse = np.zeros(length, dtype=np.double)
        impulse[impulse.shape[0] // 2] = 1.0

        self.io.save_wav(
            os.path.join(self.output_dir, "impulse.wav"),
            self.sample_rate,
            impulse,
        )

        return impulse

    def generate_sine_wave(
        self, frequency: float, length: int, window: np.ndarray = None
    ):
        """
        Generates a sine wave signal and saves it as a WAV file.

        The sine wave is generated with the specified frequency and duration.

        Parameters
        ----------
        frequency : float
            The frequency of the sine wave, in Hertz.
        length : int
            The length of the sine wave signal, in samples.
        window : np.ndarray, optional
            The window to apply to the sine wave signal. If not provided, a Gaussian window is used.

        Returns
        -------
        tuple
            A tuple containing the sine wave signal as a NumPy array and the Gaussian window used to generate it.

        """
        t = np.arange(length, dtype=np.longdouble) / self.sample_rate
        sine_wave = np.sin(2 * np.pi * frequency * t, dtype=np.longdouble)

        if window is not None:
            sine_wave = sine_wave * window

        self.io.save_wav(
            os.path.join(self.output_dir, "sine.wav"),
            self.sample_rate,
            sine_wave.astype(np.double),
        )
        return sine_wave, window
