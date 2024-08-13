import module.printer as printer
import module.plotter as plotter
import module.io as io
import time
import os
import numpy as np

CONFIG = {
    "sample_rate": 48000,
    "load_dir_sweep": "./effected/sweep",
    "output_dir": os.path.join("output_analyze_sweep", time.strftime("%Y%m%d-%H%M%S")),
}


def main():
    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    p = printer.printer(CONFIG["output_dir"])
    for key, value in CONFIG.items():
        p.print_message(f"{key}: {value}")

    p.print_message("Loading sweep...")
    _io = io.io()
    audio_path_list = []
    for root, _, files in os.walk(CONFIG["load_dir_sweep"]):
        for file in files:
            if os.path.splitext(file)[1] == ".wav":
                audio_path_list.append(os.path.join(root, file))
    audio_path_list = sorted(audio_path_list)
    p.print_message(f"audio_path_list: {audio_path_list}")
    wave_dict_list: list[plotter.AnalyzeSweepDict] = []
    for audio_path in audio_path_list:
        sample_rate, audio_data = _io.load_wav_as_mono(audio_path)
        assert (
            sample_rate == CONFIG["sample_rate"]
        ), f"sample rate mismatch: {sample_rate}"
        wave_dict_list.append(
            {
                "sweep": audio_data,
                "title": os.path.splitext(os.path.basename(audio_path))[0],
            }
        )

    p.print_message("Plotting result...")
    plot = plotter.plotter(CONFIG["output_dir"])
    for analyze_sweep_dict in wave_dict_list:
        plot.plot_mono_audio_spectrogram(
            analyze_sweep_dict["sweep"],
            CONFIG["sample_rate"],
            False,
            f"[{analyze_sweep_dict['title']}] ",
        )

    p.print_message("Done!")


if __name__ == "__main__":
    main()
