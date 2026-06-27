"""Wizard step 2: profile selection and conversion progress."""

from __future__ import annotations

import customtkinter as ctk

from encode_profiles import DEFAULT_PROFILE, PROFILE_NAMES
from gui.theme import (
    TEXT,
    TEXT_ERROR,
    TEXT_LINK,
    TEXT_OK,
    TEXT_SECONDARY,
    step_header_font,
)

DEFAULT_MIN_FREE_GB = 10.0
MIN_FREE_SLIDER_MAX_GB = 50.0


class StepConvert(ctk.CTkFrame):
    def __init__(self, master, on_back, on_start, **kwargs):
        super().__init__(master, **kwargs)
        self.on_back = on_back
        self.on_start = on_start
        self.selected_profile = ctk.StringVar(value=DEFAULT_PROFILE)
        self._profile_frames: dict[str, ctk.CTkFrame] = {}
        self._profile_space_labels: dict[str, ctk.CTkLabel] = {}
        self._scan_data: dict | None = None
        self._free_gb: float = 0.0
        self._details_visible = False
        self._log_buffer: list[str] = []
        self._is_converting = False

        self.btn_row = ctk.CTkFrame(self, fg_color="transparent")
        self.btn_row.pack(side="bottom", fill="x", padx=24, pady=16)

        self.back_btn = ctk.CTkButton(self.btn_row, text="Back", command=on_back, width=100)
        self.back_btn.pack(side="left")

        self.start_btn = ctk.CTkButton(
            self.btn_row,
            text="Start conversion",
            command=self._handle_start,
            width=160,
        )
        self.start_btn.pack(side="right")

        self.open_btn = ctk.CTkButton(
            self.btn_row,
            text="Open output folder",
            command=self._open_output,
            width=160,
            state="disabled",
        )
        self.open_btn.pack(side="right", padx=(0, 8))
        self._output_dir: str | None = None

        self.body = ctk.CTkFrame(self, fg_color="transparent")
        self.body.pack(side="top", fill="both", expand=True)

        self.header = ctk.CTkLabel(
            self.body,
            text="Step 2 of 2 — Choose quality",
            font=step_header_font(),
            text_color=TEXT,
        )
        self.header.pack(anchor="w", padx=24, pady=(24, 8))

        self.summary = ctk.CTkLabel(
            self.body, text="", wraplength=520, text_color=TEXT,
        )
        self.summary.pack(anchor="w", padx=24, pady=(0, 12))

        self.cards_frame = ctk.CTkFrame(self.body, fg_color="transparent")
        self.cards_frame.pack(fill="x", padx=24, pady=8)

        self.space_frame = ctk.CTkFrame(self.body, fg_color="transparent")
        self.space_frame.pack(fill="x", padx=24, pady=(4, 8))

        self.destination_label = ctk.CTkLabel(
            self.space_frame,
            text="",
            wraplength=520,
            anchor="w",
            text_color=TEXT,
        )
        self.destination_label.pack(anchor="w", pady=(0, 6))

        slider_row = ctk.CTkFrame(self.space_frame, fg_color="transparent")
        slider_row.pack(fill="x")

        ctk.CTkLabel(
            slider_row,
            text="Keep free after conversion:",
            anchor="w",
            text_color=TEXT,
        ).pack(side="left")

        self.min_free_value_label = ctk.CTkLabel(
            slider_row,
            text=f"{DEFAULT_MIN_FREE_GB:.0f} GB",
            width=56,
            anchor="e",
            text_color=TEXT,
        )
        self.min_free_value_label.pack(side="right")

        self.min_free_slider = ctk.CTkSlider(
            self.space_frame,
            from_=0,
            to=MIN_FREE_SLIDER_MAX_GB,
            number_of_steps=int(MIN_FREE_SLIDER_MAX_GB),
            command=self._on_min_free_changed,
        )
        self.min_free_slider.set(DEFAULT_MIN_FREE_GB)
        self.min_free_slider.pack(fill="x", pady=(4, 6))

        self.space_status_label = ctk.CTkLabel(
            self.space_frame,
            text="",
            wraplength=520,
            anchor="w",
            text_color=TEXT,
        )
        self.space_status_label.pack(anchor="w")

        self.details_frame = ctk.CTkFrame(self.body, fg_color="transparent")
        self.details_frame.pack(fill="x", padx=24, pady=(8, 0))

        self.details_toggle = ctk.CTkButton(
            self.details_frame,
            text="Show details",
            fg_color="transparent",
            text_color=TEXT_LINK,
            command=self._toggle_details,
            width=120,
        )
        self.details_toggle.pack(anchor="w")

        self.log_box = ctk.CTkTextbox(
            self.details_frame, height=160, state="disabled", text_color=TEXT,
        )

        self.status_label = ctk.CTkLabel(
            self.body, text="", wraplength=520, text_color=TEXT,
        )
        self.status_label.pack(anchor="w", padx=24, pady=(12, 4))

        self.progress = ctk.CTkProgressBar(self.body, mode="indeterminate")
        self.progress.set(0)

    def set_scan_data(self, data: dict) -> None:
        self._scan_data = data
        self._free_gb = float(data.get("free_gb", 0))
        file_count = data.get("file_count", 0)
        input_gb = data.get("input_gb", 0)
        self.summary.configure(
            text=f"Found {file_count} files · {input_gb:.1f} GB source",
        )
        self.destination_label.configure(
            text=f"Destination drive: {self._free_gb:.1f} GB free",
        )
        self._build_profile_cards(data)
        self._update_space_display()
        self._write_log_buffer()

    def _current_min_free_gb(self) -> float:
        return float(self.min_free_slider.get())

    def _on_min_free_changed(self, _value: float) -> None:
        self.min_free_value_label.configure(
            text=f"{self._current_min_free_gb():.0f} GB",
        )
        self._update_space_display()

    def _required_gb(self, profile_name: str) -> float:
        if not self._scan_data:
            return 0.0
        output_gb = self._scan_data["estimates"][profile_name]["output_gb"]
        return output_gb + self._current_min_free_gb()

    def _update_space_display(self) -> None:
        if not self._scan_data:
            return

        buffer_gb = self._current_min_free_gb()
        selected = self.selected_profile.get()
        selected_ok = False

        for name in PROFILE_NAMES:
            output_gb = self._scan_data["estimates"][name]["output_gb"]
            required_gb = output_gb + buffer_gb
            ok = self._free_gb >= required_gb
            if ok:
                text = (
                    f"Requires ~{required_gb:.1f} GB free "
                    f"({output_gb:.1f} GB output + {buffer_gb:.0f} GB buffer) — OK"
                )
                color = TEXT_OK
            else:
                shortfall = required_gb - self._free_gb
                text = (
                    f"Requires ~{required_gb:.1f} GB free "
                    f"({output_gb:.1f} GB output + {buffer_gb:.0f} GB buffer) — "
                    f"need {shortfall:.1f} GB more"
                )
                color = TEXT_ERROR

            label = self._profile_space_labels.get(name)
            if label is not None:
                label.configure(text=text, text_color=color)

            if name == selected:
                selected_ok = ok

        if selected_ok:
            required = self._required_gb(selected)
            time_display = self._scan_data["estimates"][selected]["time_display"]
            self.space_status_label.configure(
                text=(
                    f"Enough space for {selected.title()}: "
                    f"~{required:.1f} GB needed, {self._free_gb:.1f} GB available.\n"
                    f"Time to complete: ~{time_display}"
                ),
                text_color=TEXT_OK,
            )
        else:
            required = self._required_gb(selected)
            shortfall = required - self._free_gb
            self.space_status_label.configure(
                text=(
                    f"Not enough space for {selected.title()}: "
                    f"need {shortfall:.1f} GB more, or lower the free-space slider."
                ),
                text_color=TEXT_ERROR,
            )

        if not self._is_converting:
            self.start_btn.configure(state="normal" if selected_ok else "disabled")

    def _build_profile_cards(self, data: dict) -> None:
        for child in self.cards_frame.winfo_children():
            child.destroy()
        self._profile_frames.clear()
        self._profile_space_labels.clear()

        profiles = data.get("profiles", {})
        estimates = data.get("estimates", {})

        for name in PROFILE_NAMES:
            info = profiles.get(name, {})
            est = estimates.get(name, {})
            recommended = name == DEFAULT_PROFILE
            card = ctk.CTkFrame(
                self.cards_frame,
                cursor="hand2",
            )
            card.pack(fill="x", pady=6)

            title = info.get("label", name.title())
            if recommended:
                title += " (recommended)"
            radio = ctk.CTkRadioButton(
                card,
                text=title,
                variable=self.selected_profile,
                value=name,
                command=self._select_profile,
                text_color=TEXT,
            )
            radio.pack(anchor="w", padx=12, pady=(10, 0))

            desc = info.get("description", "")
            detail = (
                f"~{est.get('output_gb', 0):.1f} GB output · "
                f"~{est.get('time_display', '?')} · "
                f"{info.get('settings_summary', '')}"
            )
            detail_label = ctk.CTkLabel(
                card,
                text=f"{desc}\n{detail}",
                justify="left",
                wraplength=480,
                text_color=TEXT_SECONDARY,
                cursor="hand2",
            )
            detail_label.pack(anchor="w", padx=32, pady=(0, 4))

            space_label = ctk.CTkLabel(
                card,
                text="",
                justify="left",
                wraplength=480,
                cursor="hand2",
                text_color=TEXT,
            )
            space_label.pack(anchor="w", padx=32, pady=(0, 10))
            self._profile_space_labels[name] = space_label

            self._profile_frames[name] = card
            self._bind_profile_select(card, name, radio, detail_label, space_label)

        self._highlight_selected()

    def _bind_profile_select(self, card: ctk.CTkFrame, name: str, *widgets) -> None:
        def select(_event=None) -> None:
            self.selected_profile.set(name)
            self._select_profile()

        for widget in (card, *widgets):
            widget.bind("<Button-1>", select)

    def _select_profile(self) -> None:
        self._highlight_selected()
        self._update_space_display()

    def _highlight_selected(self) -> None:
        selected = self.selected_profile.get()
        for name, frame in self._profile_frames.items():
            if name == selected:
                frame.configure(border_width=2, border_color="#3B8ED0")
            else:
                frame.configure(border_width=0)

    def _handle_start(self) -> None:
        if self._scan_data is None:
            return
        profile = self.selected_profile.get()
        if self._free_gb < self._required_gb(profile):
            return
        self.on_start(profile, self._current_min_free_gb())

    def set_converting(self, converting: bool) -> None:
        self._is_converting = converting
        state = "disabled" if converting else "normal"
        self.back_btn.configure(state=state)
        self.min_free_slider.configure(state=state)
        for frame in self._profile_frames.values():
            for widget in frame.winfo_children():
                if isinstance(widget, ctk.CTkRadioButton):
                    widget.configure(state=state)
        if converting:
            self.start_btn.configure(state="disabled")
            self.progress.pack(fill="x", padx=24, pady=4, after=self.status_label)
            self.progress.start()
        else:
            self.progress.stop()
            self.progress.pack_forget()
            self._update_space_display()

    def set_status(self, text: str) -> None:
        self.status_label.configure(text=text)

    def set_log_buffer(self, lines: list[str]) -> None:
        self._log_buffer = list(lines)

    def append_log(self, line: str) -> None:
        self._log_buffer.append(line)
        if self._details_visible:
            self._append_log_line(line)

    def _append_log_line(self, line: str) -> None:
        self.log_box.configure(state="normal")
        self.log_box.insert("end", line + "\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")
        if "Processing file" in line or "Converting:" in line:
            self.set_status(line.strip())

    def _write_log_buffer(self) -> None:
        self.log_box.configure(state="normal")
        self.log_box.delete("1.0", "end")
        if self._log_buffer:
            self.log_box.insert("end", "\n".join(self._log_buffer) + "\n")
        self.log_box.configure(state="disabled")

    def set_done(self, exit_code: int, output_dir: str) -> None:
        self._output_dir = output_dir
        if exit_code == 0:
            self.set_status("Finished — conversion completed successfully.")
        else:
            self.set_status("Finished — some files failed. See details for more info.")
        self.open_btn.configure(state="normal")

    def _toggle_details(self) -> None:
        self._details_visible = not self._details_visible
        if self._details_visible:
            self.log_box.pack(fill="x", padx=24, pady=(0, 8))
            self.details_toggle.configure(text="Hide details")
            self._write_log_buffer()
            if not self._log_buffer:
                self.log_box.configure(state="normal")
                self.log_box.insert(
                    "end",
                    "Details will appear here during scanning and conversion.\n",
                )
                self.log_box.configure(state="disabled")
        else:
            self.log_box.pack_forget()
            self.details_toggle.configure(text="Show details")

    def _open_output(self) -> None:
        if not self._output_dir:
            return
        from gui.app import open_folder

        open_folder(self._output_dir)

    def reset_progress(self) -> None:
        self.open_btn.configure(state="disabled")
        self.set_converting(False)
