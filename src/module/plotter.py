from typing import TypedDict
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import os
import scipy.fft

mpl.rcParams["agg.path.chunksize"] = 100000


class AnalyzeDict(TypedDict):
    impulse: np.ndarray
    sine_wave: np.ndarray
    title: str


class WindowList(TypedDict):
    title: str
    window: np.ndarray


class plotter:
    def __init__(self, output_dir: str) -> None:
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def _downsample_with_indices(data, max_samples, central_samples=100):
        length = len(data)
        if length <= max_samples:
            indices = np.arange(length)
            return indices, data

        # Determine the range for the central part
        central_start = (length - central_samples) // 2
        central_end = central_start + central_samples

        # Downsample the data excluding the central part
        left_part = data[:central_start]
        right_part = data[central_end:]
        if len(left_part) + len(right_part) > max_samples - central_samples:
            factor = (len(left_part) + len(right_part)) // (
                max_samples - central_samples
            )
            left_indices = np.arange(len(left_part))[::factor]
            right_indices = np.arange(len(right_part))[::factor]
            indices = np.concatenate(
                (
                    left_indices,
                    np.arange(central_start, central_end),
                    right_indices + central_end,
                )
            )
        else:
            indices = np.concatenate(
                (
                    np.arange(len(left_part)),
                    np.arange(central_start, central_end),
                    np.arange(len(right_part)) + central_end,
                )
            )

        # Combine the left, central, and right parts
        downsampled_data = data[indices]
        return indices, downsampled_data

    def plot_window(self, window_list: list[WindowList], sample_rate: float):
        fig, ax = plt.subplots(figsize=(15, 7), layout="constrained")
        for window in window_list:
            ax.plot(window["window"], label=window["title"])
        ax.set_title("window")
        ax.legend()
        plt.savefig(os.path.join(self.output_dir, "window.png"))

    def plot_window_spectrum(self, window_list: list[WindowList], sample_rate: float):
        fig, ax = plt.subplots(figsize=(15, 7), layout="constrained")
        for window in window_list:
            window_fft = scipy.fft.fft(window["window"])
            window_fft_freq = np.fft.fftfreq(len(window_fft), 1 / sample_rate)
            fft_max = np.max(np.abs(window_fft))
            ax.plot(
                window_fft_freq,
                20 * np.log10(np.abs(window_fft) / fft_max),
                label=window["title"],
            )
        ax.set_xscale("symlog")
        ax.set_title(f"window spectrum")
        ax.grid(which="both", axis="both")
        ax.legend()

        plt.savefig(os.path.join(self.output_dir, "window_spectrum.png"))

    def plot_analysis_result(
        self,
        impulse_dict_list: list[AnalyzeDict],
        sample_rate: float,
        zoom: int = 50,
        important_freq: float = 200,
    ):

        fig, ax = plt.subplots(figsize=(20, 35), layout="constrained", nrows=6)

        for impulse_dict in impulse_dict_list:
            i = 0
            impulse = impulse_dict["impulse"]
            impulse_fft = scipy.fft.fft(impulse)
            impulse_fft_freq = np.fft.fftfreq(len(impulse_fft), 1 / sample_rate)

            plot_index_start = len(impulse) // 2 - zoom
            plot_index_end = len(impulse) // 2 + zoom
            second_max_value = np.sort(
                np.abs(impulse[plot_index_start:plot_index_end]) ** 20
            )[::-1][1]

            ax[i].plot(
                (impulse[plot_index_start:plot_index_end] ** 20)
                * np.sign(impulse[plot_index_start:plot_index_end]),
                label=impulse_dict["title"],
                linewidth=0.5,
            )
            ax[i].set_title(
                f"impulse response (center zoom) log scale, {zoom} samples ->{zoom/sample_rate:.2f}s ({1e3*zoom/sample_rate:.2f}ms)"
            )
            ax[i].set_yscale("symlog", linthresh=1e-150)
            ax[i].set_ylim(-second_max_value, second_max_value)
            yticklabels = ax[i].get_yticklabels()
            for idx in range(len(yticklabels)):
                text = yticklabels[idx].get_text()
                if "{-10^" in text:
                    yticklabels[idx].set_text("$\\mathdefault{" + text[18:-2] + "}$")
                elif "{10^" in text:
                    yticklabels[idx].set_text("$\\mathdefault{" + text[17:-2] + "}$")

                elif text == "$\\mathdefault{0}$":
                    yticklabels[idx].set_text("$\\mathdefault{-\\infty}$")
            ax[i].set_yticklabels(yticklabels)
            ax[i].legend(loc=4)
            ax[i].grid(which="both", axis="both")
            ax[i].set_ylabel("Amplitude [dB]")
            ax[i].set_xlabel("Samples")
            i += 1

            plot_index = np.where(
                (impulse_fft_freq >= 1) & (impulse_fft_freq <= sample_rate / 2)
            )
            max_value = np.max(np.abs(impulse_fft[plot_index]))
            ax[i].plot(
                impulse_fft_freq[plot_index],
                20 * np.log10(np.abs(impulse_fft[plot_index]) / max_value),
                label=impulse_dict["title"],
            )
            ax[i].set_title("impulse frequency characteristic")
            ax[i].axvline(
                important_freq,
                color="red",
                linestyle="--",
                label="important frequency",
                linewidth=0.5,
            )
            ax[i].legend()
            ax[i].grid(which="both", axis="both")
            ax[i].set_xscale("log")
            ax[i].set_xlabel("Frequency [Hz]")
            ax[i].set_ylabel("Amplitude [dB]")
            i += 1

            # plot phase responce
            rolled_impulse = np.roll(impulse, len(impulse) // 2)
            rolled_impulse_fft = scipy.fft.fft(rolled_impulse)
            impulse_phase = np.angle(rolled_impulse_fft, deg=True)
            ax[i].plot(
                impulse_fft_freq[plot_index],
                impulse_phase[plot_index],
                label=impulse_dict["title"],
            )
            ax[i].set_title("impulse phase characteristic")
            ax[i].set_ylim(-200, 200)
            ax[i].set_yticks(np.arange(-180, 181, 45))
            ax[i].axvline(
                important_freq,
                color="red",
                linestyle="--",
                label="important frequency",
                linewidth=0.5,
            )
            ax[i].legend()
            ax[i].grid(which="both", axis="both")
            ax[i].set_xscale("log")
            ax[i].set_xlabel("Frequency [Hz]")
            ax[i].set_ylabel("Phase [degree]")
            i += 1

            # sine wave
            sine_wave_fft = scipy.fft.fft(impulse_dict["sine_wave"])
            sine_wave_fft_freq = np.fft.fftfreq(len(sine_wave_fft), 1 / sample_rate)
            sine_wave_fft_max_value = np.max(np.abs(sine_wave_fft[plot_index]))
            ax[i].plot(
                sine_wave_fft_freq[plot_index],
                20
                * np.log10(np.abs(sine_wave_fft[plot_index]) / sine_wave_fft_max_value),
                label=impulse_dict["title"],
            )
            ax[i].set_title("distortion frequency characteristic")
            ax[i].axvline(
                important_freq,
                color="red",
                linestyle="--",
                label="important frequency",
                linewidth=0.5,
            )
            ax[i].legend()
            ax[i].grid(which="both", axis="both")
            ax[i].set_xscale("log")
            ax[i].set_xlabel("Frequency [Hz]")
            ax[i].set_ylabel("Amplitude [dB]")
            i += 1

            # zoom in
            plot_index = np.where(
                (impulse_fft_freq > important_freq / 2)
                & (impulse_fft_freq < important_freq * 2)
                & (impulse_fft_freq >= 1)
                & (impulse_fft_freq <= sample_rate / 2)
            )
            ax[i].plot(
                impulse_fft_freq[plot_index],
                20 * np.log10(np.abs(impulse_fft[plot_index]) / max_value),
                label=impulse_dict["title"],
            )
            ax[i].set_title("impulse frequency characteristic (zoom in)")
            ax[i].axvline(
                important_freq,
                color="red",
                linestyle="--",
                label="important frequency",
                linewidth=0.5,
            )
            ax[i].legend()
            ax[i].grid(which="both", axis="both")
            ax[i].set_xscale("log")
            ax[i].set_xlabel("Frequency [Hz]")
            ax[i].set_ylabel("Amplitude [dB]")
            i += 1

            ax[i].plot(
                impulse_fft_freq[plot_index],
                20 * np.log10(np.abs(impulse_fft[plot_index]) / max_value),
                label=impulse_dict["title"],
            )
            ax[i].set_title("impulse frequency characteristic (zoom in)")
            ax[i].set_ylim(-24, 1)
            ax[i].set_yticks(np.arange(-24, 1, 3))
            ax[i].axvline(
                important_freq,
                color="red",
                linestyle="--",
                label="important frequency",
                linewidth=0.5,
            )
            ax[i].legend()
            ax[i].grid(which="both", axis="both")
            ax[i].set_xscale("log")
            ax[i].set_xlabel("Frequency [Hz]")
            ax[i].set_ylabel("Amplitude [dB]")
            i += 1

        plt.savefig(os.path.join(self.output_dir, "impulse_freq_characteristic.png"))
        plt.savefig(os.path.join(self.output_dir, "impulse_freq_characteristic.pdf"))
