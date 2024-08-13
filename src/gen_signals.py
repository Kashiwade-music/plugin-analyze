import module.generator as generator
import module.printer as printer
import module.plotter as plotter
import os
import time
import module.windows as windows
from scipy.signal import windows as scipy_windows


CONFIG = {
    "sample_rate": 48000,
    "signal_length": 2**22,
    "sine_wave_freq": 1000,
    "sine_wave_amplitude_dBFS": -6,
    "sweep_start_freq": 1,
    "sweep_end_freq": 23000,
    "sweep_is_log_scale": False,
    "sweep_amplitude_dBFS": -6,
    "should_apply_window_to_sine_wave": True,
    "output_dir": os.path.join("output_signals", time.strftime("%Y%m%d-%H%M%S")),
}


def main():
    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    p = printer.printer(CONFIG["output_dir"])

    for key, value in CONFIG.items():
        p.print_message(f"{key}: {value}")

    gen = generator.generator(CONFIG["sample_rate"], CONFIG["output_dir"])

    p.print_message("Generating impulse...")
    gen.generate_impulse(CONFIG["signal_length"])

    p.print_message("Generating sine wave...")
    _, window = gen.generate_sine_wave(
        CONFIG["sine_wave_freq"],
        CONFIG["signal_length"],
        CONFIG["sine_wave_amplitude_dBFS"],
        window=(
            windows.gaussian_longdouble(CONFIG["signal_length"], 200000)
            # scipy_windows.nuttall(CONFIG["signal_length"])
            # * scipy_windows.kaiser(CONFIG["signal_length"], 20)
            # scipy_windows.chebwin(CONFIG["signal_length"], 400)
            if CONFIG["should_apply_window_to_sine_wave"]
            else None
        ),
    )

    if CONFIG["should_apply_window_to_sine_wave"]:
        p.print_message("Plotting window...")
        plot = plotter.plotter(CONFIG["output_dir"])
        plot.plot_window(
            [{"title": "default window", "window": window}], CONFIG["sample_rate"]
        )
        plot.plot_window_spectrum(
            [{"title": "default window", "window": window}], CONFIG["sample_rate"]
        )

    p.print_message("Generating sweep...")
    gen.generate_sweep_up(
        CONFIG["signal_length"],
        CONFIG["sweep_start_freq"],
        CONFIG["sweep_end_freq"],
        CONFIG["sweep_amplitude_dBFS"],
        log_scale=CONFIG["sweep_is_log_scale"],
    )

    p.print_message("Done!")


if __name__ == "__main__":
    main()
