"""Main CustomTkinter wizard application."""

from __future__ import annotations

import platform
import subprocess
import threading
from pathlib import Path
from queue import Empty, Queue

import customtkinter as ctk

from gui.ffmpeg_gate import ffmpeg_available, ffmpeg_install_hint
from gui.step_convert import StepConvert
from gui.step_folders import StepFolders
from gui.theme import TEXT, apply_widget_styles
from gui.workers import worker_convert, worker_scan


def open_folder(path: str) -> None:
    system = platform.system()
    if system == "Darwin":
        subprocess.run(["open", path], check=False)
    elif system == "Windows":
        subprocess.run(["explorer", path], check=False)
    else:
        subprocess.run(["xdg-open", path], check=False)


class TranscodeApp(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        self.title("HEVC Video Converter")
        self.geometry("640x720")
        self.minsize(560, 600)

        ctk.set_appearance_mode("system")
        ctk.set_default_color_theme("blue")
        apply_widget_styles()

        self.queue: Queue = Queue()
        self._log_buffer: list[str] = []
        self._input_dir: Path | None = None
        self._output_dir: Path | None = None
        self._manifest_path: str | None = None
        self._worker: threading.Thread | None = None

        if not ffmpeg_available():
            self._show_ffmpeg_gate()
            return

        self.container = ctk.CTkFrame(self, fg_color="transparent")
        self.container.pack(fill="both", expand=True)

        self.step_folders = StepFolders(self.container, on_next=self._start_scan)
        self.step_convert = StepConvert(
            self.container,
            on_back=self._show_folders,
            on_start=self._start_convert,
        )

        self._show_folders()
        self.after(100, self._poll_queue)

    def _show_ffmpeg_gate(self) -> None:
        frame = ctk.CTkFrame(self)
        frame.pack(fill="both", expand=True, padx=24, pady=24)

        ctk.CTkLabel(
            frame,
            text="FFmpeg required",
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color=TEXT,
        ).pack(anchor="w", pady=(0, 12))

        ctk.CTkLabel(
            frame,
            text=(
                "This app needs FFmpeg and ffprobe installed and available on your PATH.\n\n"
                + ffmpeg_install_hint()
            ),
            justify="left",
            wraplength=520,
            text_color=TEXT,
        ).pack(anchor="w")

        ctk.CTkButton(frame, text="Quit", command=self.destroy, width=100).pack(
            anchor="e", pady=24,
        )

    def _show_folders(self) -> None:
        self.step_convert.pack_forget()
        self.step_folders.pack(fill="both", expand=True)
        if self._input_dir and self._output_dir:
            self.step_folders.input_dir = self._input_dir
            self.step_folders.output_dir = self._output_dir
            self.step_folders._source_var.set(str(self._input_dir))
            self.step_folders._dest_var.set(str(self._output_dir))
            self.step_folders._update_next_state()

    def _show_convert(self) -> None:
        self.step_folders.pack_forget()
        self.step_convert.pack(fill="both", expand=True)

    def _start_scan(self, input_dir: Path, output_dir: Path) -> None:
        self._input_dir = input_dir
        self._output_dir = output_dir
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            self.step_folders.set_busy(False, f"Cannot create output folder: {exc}")
            return

        self.step_folders.set_busy(True, "Scanning…")
        self._log_buffer.clear()
        self._worker = threading.Thread(
            target=worker_scan,
            args=(self.queue, input_dir, output_dir),
            daemon=True,
        )
        self._worker.start()

    def _start_convert(self, profile_name: str, min_free_gb: float) -> None:
        if not self._manifest_path or not self._output_dir:
            return
        self.step_convert.set_converting(True)
        self.step_convert.set_status("Preparing conversion…")
        self._worker = threading.Thread(
            target=worker_convert,
            args=(
                self.queue,
                self._manifest_path,
                profile_name,
                self._output_dir,
                min_free_gb,
            ),
            daemon=True,
        )
        self._worker.start()

    def _poll_queue(self) -> None:
        try:
            while True:
                event, payload = self.queue.get_nowait()
                self._handle_event(event, payload)
        except Empty:
            pass
        self.after(100, self._poll_queue)

    def _handle_event(self, event: str, payload) -> None:
        if event == "status":
            if self.step_convert.winfo_ismapped():
                self.step_convert.set_status(str(payload))
            elif self.step_folders.winfo_ismapped():
                self.step_folders.set_busy(True, str(payload))
        elif event == "log":
            self._log_buffer.append(str(payload))
            if self.step_convert.winfo_ismapped():
                self.step_convert.append_log(str(payload))
        elif event == "scan_progress":
            if self.step_folders.winfo_ismapped():
                self.step_folders.set_scan_progress(
                    int(payload["checked"]),
                    int(payload["found"]),
                )
        elif event == "scan_done":
            self.step_folders.set_busy(False, "")
            data = payload
            if data.get("file_count", 0) == 0:
                self.step_folders.set_busy(
                    False,
                    "Everything here is already converted. Pick different folders.",
                )
                return
            self._manifest_path = data["manifest_path"]
            self.step_convert.set_log_buffer(self._log_buffer)
            self.step_convert.set_scan_data(data)
            self._show_convert()
        elif event == "space_fail":
            self.step_convert.set_converting(False)
            self.step_convert.set_status(str(payload))
            self._show_space_dialog(str(payload))
        elif event == "convert_done":
            self.step_convert.set_converting(False)
            info = payload
            self.step_convert.set_done(info["exit_code"], info["output_dir"])
        elif event == "error":
            if self.step_convert.winfo_ismapped():
                self.step_convert.set_converting(False)
                self.step_convert.set_status(f"Error: {payload}")
            else:
                self.step_folders.set_busy(False, f"Error: {payload}")

    def _show_space_dialog(self, message: str) -> None:
        dialog = ctk.CTkToplevel(self)
        dialog.title("Not enough disk space")
        dialog.geometry("480x220")
        dialog.transient(self)
        dialog.grab_set()

        ctk.CTkLabel(
            dialog,
            text=message,
            wraplength=440,
            justify="left",
            text_color=TEXT,
        ).pack(padx=20, pady=20, anchor="w")
        ctk.CTkButton(dialog, text="OK", command=dialog.destroy, width=80).pack(
            padx=20, pady=(0, 20), anchor="e",
        )


def run_app() -> None:
    app = TranscodeApp()
    app.mainloop()
