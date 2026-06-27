"""Wizard step 1: source and destination folders."""

from __future__ import annotations

from pathlib import Path
from tkinter import filedialog

import customtkinter as ctk

from gui.theme import TEXT, TEXT_MUTED


class StepFolders(ctk.CTkFrame):
    def __init__(self, master, on_next, **kwargs):
        super().__init__(master, **kwargs)
        self.on_next = on_next
        self.input_dir: Path | None = None
        self.output_dir: Path | None = None

        ctk.CTkLabel(
            self,
            text="Step 1 of 2 — Choose folders",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color=TEXT,
        ).pack(anchor="w", padx=24, pady=(24, 8))

        ctk.CTkLabel(
            self,
            text="Pick where your videos are and where converted files should be saved.",
            wraplength=520,
            text_color=TEXT,
        ).pack(anchor="w", padx=24, pady=(0, 16))

        self._source_var = ctk.StringVar(value="No folder selected")
        self._dest_var = ctk.StringVar(value="No folder selected")

        self._add_folder_row("Source folder", self._source_var, self._pick_source)
        self._add_folder_row("Save to folder", self._dest_var, self._pick_dest)

        self.status_label = ctk.CTkLabel(self, text="", text_color=TEXT_MUTED)
        self.status_label.pack(anchor="w", padx=24, pady=(8, 0))

        self.progress = ctk.CTkProgressBar(self, mode="indeterminate")
        self.progress.set(0)

        self.next_btn = ctk.CTkButton(
            self,
            text="Next",
            command=self._handle_next,
            state="disabled",
            width=140,
        )
        self.next_btn.pack(anchor="e", padx=24, pady=24)

    def _add_folder_row(self, label: str, var: ctk.StringVar, command) -> None:
        frame = ctk.CTkFrame(self, fg_color="transparent")
        frame.pack(fill="x", padx=24, pady=8)
        ctk.CTkLabel(frame, text=label, width=120, anchor="w", text_color=TEXT).pack(
            side="left",
        )
        ctk.CTkLabel(
            frame, textvariable=var, anchor="w", text_color=TEXT,
        ).pack(
            side="left", fill="x", expand=True, padx=(8, 8),
        )
        ctk.CTkButton(frame, text="Browse…", width=100, command=command).pack(side="right")

    def _pick_source(self) -> None:
        path = filedialog.askdirectory(title="Select source folder")
        if path:
            self.input_dir = Path(path).resolve()
            self._source_var.set(str(self.input_dir))
            self._update_next_state()

    def _pick_dest(self) -> None:
        path = filedialog.askdirectory(title="Select destination folder")
        if path:
            self.output_dir = Path(path).resolve()
            self._dest_var.set(str(self.output_dir))
            self._update_next_state()

    def _update_next_state(self) -> None:
        ready = self.input_dir is not None and self.output_dir is not None
        self.next_btn.configure(state="normal" if ready else "disabled")
        self.status_label.configure(text="")

    def _handle_next(self) -> None:
        if self.input_dir is None or self.output_dir is None:
            return
        if not self.input_dir.is_dir():
            self.status_label.configure(text="Source folder does not exist.")
            return
        self.on_next(self.input_dir, self.output_dir)

    def set_busy(self, busy: bool, message: str = "") -> None:
        self.next_btn.configure(state="disabled" if busy else "normal")
        self.status_label.configure(text=message)
        if busy:
            self.progress.pack(fill="x", padx=24, pady=(4, 0), after=self.status_label)
            self.progress.start()
        else:
            self.progress.stop()
            self.progress.pack_forget()

    def set_scan_progress(self, checked: int, found: int) -> None:
        self.status_label.configure(
            text=(
                f"Scanning media files… {checked:,} items checked"
                + (f", {found:,} to convert" if found else "")
            ),
        )

    def get_paths(self) -> tuple[Path | None, Path | None]:
        return self.input_dir, self.output_dir
