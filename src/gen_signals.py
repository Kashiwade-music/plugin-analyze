import module.generator as generator
import module.printer as printer
import module.plotter as plotter
import os
import time

CONFIG = {
    "sample_rate": 48000,
    "signal_length": 2**22,
    "sine_wave_freq": 1000,
    "output_dir": os.path.join("output_signals", time.strftime("%Y%m%d-%H%M%S")),
}


def main():
    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    p = printer.printer(CONFIG["output_dir"])
    p.print_message(f"sample_rate: {CONFIG['sample_rate']}")
    p.print_message(f"signal_length: {CONFIG['signal_length']}")
    p.print_message(f"sine_wave_freq: {CONFIG['sine_wave_freq']}")
    p.print_message(f"output_dir: '{CONFIG['output_dir']}'")

    gen = generator.generator(CONFIG["sample_rate"], CONFIG["output_dir"])

    p.print_message("Generating impulse...")
    gen.generate_impulse(CONFIG["signal_length"])

    p.print_message("Generating sine wave...")
    _, window = gen.generate_sine_wave(
        CONFIG["sine_wave_freq"], CONFIG["signal_length"]
    )

    p.print_message("Plotting window...")
    plot = plotter.plotter(CONFIG["output_dir"])
    plot.plot_window(
        [{"title": "default window", "window": window}], CONFIG["sample_rate"]
    )
    plot.plot_window_spectrum(
        [{"title": "default window", "window": window}], CONFIG["sample_rate"]
    )

    p.print_message("Done!")


if __name__ == "__main__":
    main()
