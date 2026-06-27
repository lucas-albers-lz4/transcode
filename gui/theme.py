"""GUI text colors — higher contrast, same sizes as defaults."""

import customtkinter as ctk

# (light mode, dark mode)
TEXT = ("#0D0D0D", "#F0F0F0")
TEXT_SECONDARY = ("#1F1F1F", "#D8D8D8")
TEXT_MUTED = ("#333333", "#C0C0C0")
TEXT_OK = ("#0F5132", "#5DDB89")
TEXT_ERROR = ("#842029", "#F08080")
TEXT_LINK = ("#0D0D0D", "#E8E8E8")

STEP_HEADER_SIZE = 20
HEADER_FONT = "Georgia"


def step_header_font(size: int = STEP_HEADER_SIZE) -> ctk.CTkFont:
    """Bold font for wizard step titles."""
    return ctk.CTkFont(family=HEADER_FONT, size=size, weight="bold")


def apply_widget_styles() -> None:
    """Apply app-wide CustomTkinter widget styling."""
    ctk.ThemeManager.theme["CTkButton"]["corner_radius"] = 0
