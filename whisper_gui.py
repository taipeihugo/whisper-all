"""Simple GUI for batch-transcribing audio files with Whisper.

Run this script from the repository root (where `whisper` package is located):

    python whisper_gui.py

The GUI lets you select multiple audio files, choose a model, and write outputs in
TXT/SRT/VTT/TSV/JSON formats.

Requirements:
- Python 3.8+
- whisper dependencies (torch, etc.)

"""

import os
import threading
import tkinter as tk
import tkinter.filedialog as fd
import tkinter.messagebox as mb

import whisper
from whisper.utils import get_writer

AUDIO_FILETYPES = [
    ("Audio files", "*.mp4 *.wav *.mp3 *.flac *.m4a *.aac *.ogg *.opus"),
    ("All files", "*.*"),
]

MODEL_CHOICES = ["tiny", "base", "small", "medium", "large", "turbo"]
OUTPUT_FORMATS = ["txt", "srt", "vtt", "tsv", "json"]


class WhisperGui:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Whisper Batch Transcriber")

        frame = tk.Frame(root, padx=10, pady=10)
        frame.pack(fill=tk.BOTH, expand=True)

        tk.Label(frame, text="Selected audio files:").grid(row=0, column=0, sticky="w")
        self.file_listbox = tk.Listbox(frame, width=80, height=8, selectmode=tk.EXTENDED)
        self.file_listbox.grid(row=1, column=0, columnspan=3, sticky="nsew", pady=(0, 8))

        self.select_btn = tk.Button(frame, text="Select Audio Files…", command=self.select_files)
        self.select_btn.grid(row=2, column=0, sticky="w")

        tk.Label(frame, text="Model:").grid(row=3, column=0, sticky="w", pady=(8, 0))
        self.model_var = tk.StringVar(value="turbo")
        self.model_menu = tk.OptionMenu(frame, self.model_var, *MODEL_CHOICES)
        self.model_menu.grid(row=3, column=1, sticky="w", pady=(8, 0))

        tk.Label(frame, text="Output format:").grid(row=4, column=0, sticky="w", pady=(4, 0))
        self.format_var = tk.StringVar(value="txt")
        self.format_menu = tk.OptionMenu(frame, self.format_var, *OUTPUT_FORMATS)
        self.format_menu.grid(row=4, column=1, sticky="w", pady=(4, 0))

        tk.Label(frame, text="Output directory:").grid(row=5, column=0, sticky="w", pady=(4, 0))
        self.output_dir_var = tk.StringVar(value=os.getcwd())
        self.output_dir_entry = tk.Entry(frame, textvariable=self.output_dir_var, width=60)
        self.output_dir_entry.grid(row=6, column=0, columnspan=2, sticky="we", pady=(0, 8))
        self.output_dir_btn = tk.Button(frame, text="Browse…", command=self.select_output_dir)
        self.output_dir_btn.grid(row=6, column=2, sticky="w", padx=(8, 0))

        self.start_btn = tk.Button(frame, text="Start transcription", command=self.start)
        self.start_btn.grid(row=7, column=0, columnspan=3, pady=(0, 10))

        tk.Label(frame, text="Log:").grid(row=8, column=0, sticky="w")
        self.log_text = tk.Text(frame, width=80, height=12, state=tk.DISABLED)
        self.log_text.grid(row=9, column=0, columnspan=3, sticky="nsew")

        frame.grid_rowconfigure(9, weight=1)
        frame.grid_columnconfigure(0, weight=1)

    def log(self, message: str):
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def select_files(self):
        paths = fd.askopenfilenames(title="Select audio files", filetypes=AUDIO_FILETYPES)
        if paths:
            self.file_listbox.delete(0, tk.END)
            for p in paths:
                self.file_listbox.insert(tk.END, p)

    def select_output_dir(self):
        path = fd.askdirectory(title="Select output directory", initialdir=self.output_dir_var.get())
        if path:
            self.output_dir_var.set(path)

    def start(self):
        files = list(self.file_listbox.get(0, tk.END))
        if not files:
            mb.showwarning("No files", "Please select one or more audio files to transcribe.")
            return

        output_dir = self.output_dir_var.get().strip() or os.getcwd()
        os.makedirs(output_dir, exist_ok=True)

        self.start_btn.configure(state=tk.DISABLED)
        self.log(f"Loading model '{self.model_var.get()}'... (this may take a while)")

        thread = threading.Thread(
            target=self._run_transcribe,
            args=(files, self.model_var.get(), output_dir, self.format_var.get()),
            daemon=True,
        )
        thread.start()

    def _run_transcribe(self, files, model_name, output_dir, output_format):
        try:
            model = whisper.load_model(model_name)
        except Exception as e:
            self.log(f"Error loading model: {e}")
            self.start_btn.configure(state=tk.NORMAL)
            return

        writer = get_writer(output_format, output_dir)

        for i, audio_path in enumerate(files, start=1):
            self.log(f"[{i}/{len(files)}] Transcribing: {audio_path}")
            try:
                result = model.transcribe(audio_path)
                writer(result, audio_path)
                self.log(f"  -> saved as {os.path.splitext(os.path.basename(audio_path))[0]}.{output_format}")
            except Exception as e:
                self.log(f"  ✗ failed: {e}")

        self.log("Done.")
        self.start_btn.configure(state=tk.NORMAL)


def main():
    root = tk.Tk()
    app = WhisperGui(root)
    root.mainloop()


if __name__ == "__main__":
    main()
