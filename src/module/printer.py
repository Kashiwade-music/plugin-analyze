from rich import print
import os
import time


class printer:
    def __init__(self, output_filepath: str):
        self.output_filepath = os.path.join(output_filepath, "output.txt")
        os.makedirs(output_filepath, exist_ok=True)

    def print_message(self, message):
        with open(self.output_filepath, "a") as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")
        print(
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}",
        )
