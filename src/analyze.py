import module.printer as printer
import module.plotter as plotter
import module.io as io
import time
import os
import numpy as np

CONFIG = {
    "sample_rate": 48000,
    "load_dir_impulse": "./effected/impulse",
    "load_dir_sin": "./effected/sin",
    "fft_size": 2**23,
    "plot_zoom": 3000,
    "plot_important_freq": 200,
    "output_dir": os.path.join("output_analyze", time.strftime("%Y%m%d-%H%M%S")),
}


def main():
    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    p = printer.printer(CONFIG["output_dir"])
    p.print_message(f"sample_rate: {CONFIG['sample_rate']}")
    p.print_message(f"load_dir_impulse: '{CONFIG['load_dir_impulse']}'")
    p.print_message(f"load_dir_sin: '{CONFIG['load_dir_sin']}'")
    p.print_message(f"fft_size: {CONFIG['fft_size']}")
    p.print_message(f"output_dir: '{CONFIG['output_dir']}'")

    p.print_message("Loading impulse...")
    _io = io.io()
    audio_path_list = []
    for root, _, files in os.walk(CONFIG["load_dir_impulse"]):
        for file in files:
            if os.path.splitext(file)[1] == ".wav":
                audio_path_list.append(os.path.join(root, file))
    audio_path_list = sorted(audio_path_list)
    p.print_message(f"audio_path_list: {audio_path_list}")
    wave_dict_list: list[plotter.AnalyzeDict] = []
    for audio_path in audio_path_list:
        sample_rate, audio_data = _io.load_wav_as_mono(audio_path)
        assert (
            sample_rate == CONFIG["sample_rate"]
        ), f"sample rate mismatch: {sample_rate}"
        pad_length = CONFIG["fft_size"] - audio_data.shape[0]
        if pad_length > 0:
            audio_data = np.pad(audio_data, (pad_length // 2, pad_length // 2))
        audio_data /= np.max(np.abs(audio_data))
        wave_dict_list.append(
            {
                "impulse": audio_data,
                "title": os.path.splitext(os.path.basename(audio_path))[0],
            }
        )

    p.print_message("Loading sine...")
    audio_path_list = []
    for root, _, files in os.walk(CONFIG["load_dir_sin"]):
        for file in files:
            if os.path.splitext(file)[1] == ".wav":
                audio_path_list.append(os.path.join(root, file))
    audio_path_list = sorted(audio_path_list)
    p.print_message(f"audio_path_list: {audio_path_list}")
    for idx, audio_path in enumerate(audio_path_list):
        sample_rate, audio_data = _io.load_wav_as_mono(audio_path)
        assert (
            sample_rate == CONFIG["sample_rate"]
        ), f"sample rate mismatch: {sample_rate}"
        pad_length = CONFIG["fft_size"] - audio_data.shape[0]
        if pad_length > 0:
            audio_data = np.pad(audio_data, (pad_length // 2, pad_length // 2))
        audio_data /= np.max(np.abs(audio_data))
        wave_dict_list[idx]["sine_wave"] = audio_data

        # check title is same
        assert (
            wave_dict_list[idx]["title"]
            == os.path.splitext(os.path.basename(audio_path))[0]
        ), f"title mismatch: {wave_dict_list[idx]['title']}"

    p.print_message("Plotting result...")
    plot = plotter.plotter(CONFIG["output_dir"])
    plot.plot_analysis_result(
        wave_dict_list,
        CONFIG["sample_rate"],
        zoom=CONFIG["plot_zoom"],
        important_freq=CONFIG["plot_important_freq"],
    )

    p.print_message("Done!")


if __name__ == "__main__":
    main()
