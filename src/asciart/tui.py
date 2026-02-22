from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable

from PIL import Image
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import (
    Button, DirectoryTree, Footer, Header, Input, Label, Select, Static, Switch,
)
from textual.timer import Timer
from textual_slider import Slider

from asciart.core.engine import convert
from asciart.models import ColorMode, ConvertOptions, DitherMode, MatchMode, Mode
from asciart.core.ramps import RAMPS
from asciart.presets import PRESETS
from asciart.renderers.text import render_text
from asciart.renderers.html import render_html
from asciart.renderers.image import render_png

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff", ".tif"}


class ImageDirectoryTree(DirectoryTree):
    """DirectoryTree filtered to only show image files."""

    def filter_paths(self, paths: Iterable[Path]) -> Iterable[Path]:
        return [
            p for p in paths
            if p.is_dir() or p.suffix.lower() in IMAGE_EXTENSIONS
        ]


class FilePickerScreen(Screen[str]):
    """Screen for selecting an image file via browsing or path input."""

    BINDINGS = [("escape", "cancel", "Cancel")]

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="picker-container"):
            yield Label("Drop or paste an image path:", classes="section-label")
            yield Input(
                placeholder="Paste or type image path here...",
                id="path-input",
            )
            yield Label("Or browse for an image:", classes="section-label")
            yield ImageDirectoryTree(str(Path.cwd()), id="file-tree")
            with Horizontal(id="picker-buttons"):
                yield Button("Open", id="open-btn", variant="primary")
                yield Button("Cancel", id="cancel-btn", variant="default")
        yield Footer()

    def on_directory_tree_file_selected(self, event: DirectoryTree.FileSelected) -> None:
        """When a file is clicked in the tree, put its path in the input."""
        self.query_one("#path-input", Input).value = str(event.path)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "open-btn":
            self._try_open()
        elif event.button.id == "cancel-btn":
            self.app.exit()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Enter key in the path input opens the file."""
        self._try_open()

    def _try_open(self) -> None:
        path_str = self.query_one("#path-input", Input).value.strip()
        if not path_str:
            return
        # Strip quotes that drag-and-drop sometimes adds
        path_str = path_str.strip("'\"")
        p = Path(path_str).expanduser()
        if not p.exists():
            self.query_one("#path-input", Input).value = f"File not found: {path_str}"
            return
        if p.suffix.lower() not in IMAGE_EXTENSIONS:
            self.query_one("#path-input", Input).value = f"Not an image: {p.name}"
            return
        self.dismiss(str(p))

    def action_cancel(self) -> None:
        self.app.exit()


class AsciiArtApp(App):
    CSS_PATH = "tui.tcss"
    TITLE = "asciart - interactive mode"
    BINDINGS = [("q", "quit", "Quit")]

    def __init__(self, image_path: str | None = None, **kwargs):
        super().__init__(**kwargs)
        self.image_path = image_path
        self._image: Image.Image | None = None
        self._debounce_timer: Timer | None = None
        self._options = ConvertOptions(width=80)

    def compose(self) -> ComposeResult:
        yield Header()

        with Horizontal():
            # Left: Preview
            with VerticalScroll(id="preview-container"):
                yield Static("Loading...", id="preview")

            # Right: Controls
            with VerticalScroll(id="controls"):
                # Presets
                yield Label("Presets", classes="section-label")
                with Horizontal(id="preset-bar"):
                    for name in PRESETS:
                        yield Button(name.capitalize(), id=f"preset-{name}", variant="default")

                # Mode
                yield Label("Mode", classes="section-label")
                yield Select(
                    [(m.value, m.value) for m in Mode],
                    value=Mode.ASCII.value,
                    id="mode-select",
                )

                # Character Ramp
                yield Label("Chars", classes="section-label")
                yield Select(
                    [(name, name) for name in RAMPS],
                    value="standard",
                    id="chars-select",
                )

                # Dimensions
                yield Label("Width", classes="section-label")
                yield Slider(min=20, max=300, value=80, step=5, id="width-slider")

                yield Label("Font Ratio", classes="section-label")
                yield Slider(min=20, max=100, value=50, step=5, id="ratio-slider")

                # Image adjustments
                yield Label("--- Image ---", classes="section-label")

                yield Label("Brightness", classes="control-label")
                yield Slider(min=-100, max=100, value=0, step=5, id="brightness-slider")

                yield Label("Contrast", classes="control-label")
                yield Slider(min=10, max=300, value=100, step=5, id="contrast-slider")

                yield Label("Saturation", classes="control-label")
                yield Slider(min=0, max=300, value=100, step=5, id="saturation-slider")

                yield Label("Sharpness", classes="control-label")
                yield Slider(min=0, max=300, value=100, step=5, id="sharpness-slider")

                # Rendering
                yield Label("--- Rendering ---", classes="section-label")

                yield Label("Dither", classes="section-label")
                yield Select(
                    [(d.value, d.value) for d in DitherMode],
                    value=DitherMode.NONE.value,
                    id="dither-select",
                )

                yield Label("Color", classes="section-label")
                yield Select(
                    [(c.value, c.value) for c in ColorMode],
                    value=ColorMode.TRUECOLOR.value,
                    id="color-select",
                )

                yield Label("Match Mode", classes="section-label")
                yield Select(
                    [(m.value, m.value) for m in MatchMode],
                    value=MatchMode.BRIGHTNESS.value,
                    id="match-select",
                )

                with Horizontal(classes="control-row"):
                    yield Label("CLAHE")
                    yield Switch(id="clahe-switch", value=False)

                with Horizontal(classes="control-row"):
                    yield Label("Edge Detection")
                    yield Switch(id="edge-switch", value=False)

                with Horizontal(classes="control-row"):
                    yield Label("Invert")
                    yield Switch(id="invert-switch", value=False)

                # Export
                yield Label("--- Export ---", classes="section-label")
                with Horizontal(classes="control-row"):
                    yield Button("Save TXT", id="save-txt")
                    yield Button("Save HTML", id="save-html")
                with Horizontal(classes="control-row"):
                    yield Button("Save PNG", id="save-png")

        yield Static("Loading...", id="status-bar")
        yield Footer()

    def on_mount(self) -> None:
        if self.image_path is None:
            self.push_screen(FilePickerScreen(), callback=self._on_file_picked)
        else:
            self._load_image(self.image_path)

    def _on_file_picked(self, path: str | None) -> None:
        """Called when FilePickerScreen dismisses with a path."""
        if path is None:
            self.exit()
            return
        self.image_path = path
        self._load_image(path)

    def _load_image(self, path: str) -> None:
        try:
            self._image = Image.open(path)
        except Exception as e:
            self.query_one("#preview", Static).update(f"Error loading image: {e}")
            return
        self.query_one("#preview", Static).update(f"Loaded: {path}")
        self._refresh_preview()

    def _build_options(self) -> ConvertOptions:
        return ConvertOptions(
            width=self.query_one("#width-slider", Slider).value,
            mode=Mode(self.query_one("#mode-select", Select).value),
            chars=str(self.query_one("#chars-select", Select).value),
            color=ColorMode(self.query_one("#color-select", Select).value),
            dither=DitherMode(self.query_one("#dither-select", Select).value),
            invert=self.query_one("#invert-switch", Switch).value,
            brightness=float(self.query_one("#brightness-slider", Slider).value),
            contrast=self.query_one("#contrast-slider", Slider).value / 100.0,
            saturation=self.query_one("#saturation-slider", Slider).value / 100.0,
            sharpness=self.query_one("#sharpness-slider", Slider).value / 100.0,
            font_ratio=self.query_one("#ratio-slider", Slider).value / 100.0,
            edge_detection=self.query_one("#edge-switch", Switch).value,
            match_mode=MatchMode(self.query_one("#match-select", Select).value),
            clahe=self.query_one("#clahe-switch", Switch).value,
        )

    def _refresh_preview(self) -> None:
        if self._image is None:
            return
        start = time.perf_counter()
        self._options = self._build_options()
        art = convert(self._image, self._options)
        rendered = render_text(art)
        elapsed = (time.perf_counter() - start) * 1000
        self.query_one("#preview", Static).update(rendered)
        from asciart.accel import get_backend
        backend = get_backend()
        status = (
            f"{art.width}x{art.height} | {self._options.mode.value} "
            f"| {self._options.match_mode.value} | {backend.name} | {elapsed:.0f}ms"
        )
        self.query_one("#status-bar", Static).update(status)

    def _schedule_refresh(self) -> None:
        """Debounced refresh: wait 100ms after last change."""
        if self._debounce_timer is not None:
            self._debounce_timer.stop()
        self._debounce_timer = self.set_timer(0.1, self._refresh_preview)

    def on_slider_changed(self, event: Slider.Changed) -> None:
        self._schedule_refresh()

    def on_select_changed(self, event: Select.Changed) -> None:
        self._schedule_refresh()

    def on_switch_changed(self, event: Switch.Changed) -> None:
        self._schedule_refresh()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        btn_id = event.button.id or ""

        # Preset buttons
        if btn_id.startswith("preset-"):
            preset_name = btn_id.replace("preset-", "")
            preset = PRESETS.get(preset_name)
            if preset:
                self._apply_preset(preset)
            return

        # Export buttons
        if btn_id == "save-txt":
            self._save_export("txt")
        elif btn_id == "save-html":
            self._save_export("html")
        elif btn_id == "save-png":
            self._save_export("png")

    def _apply_preset(self, preset: ConvertOptions) -> None:
        """Snap all sliders/selects to preset values."""
        self.query_one("#mode-select", Select).value = preset.mode.value
        ramp_name = next((k for k, v in RAMPS.items() if v == preset.chars), "standard")
        self.query_one("#chars-select", Select).value = ramp_name
        self.query_one("#width-slider", Slider).value = preset.width
        self.query_one("#brightness-slider", Slider).value = int(preset.brightness)
        self.query_one("#contrast-slider", Slider).value = int(preset.contrast * 100)
        self.query_one("#saturation-slider", Slider).value = int(preset.saturation * 100)
        self.query_one("#sharpness-slider", Slider).value = int(preset.sharpness * 100)
        self.query_one("#ratio-slider", Slider).value = int(preset.font_ratio * 100)
        self.query_one("#dither-select", Select).value = preset.dither.value
        self.query_one("#color-select", Select).value = preset.color.value
        self.query_one("#edge-switch", Switch).value = preset.edge_detection
        self.query_one("#invert-switch", Switch).value = preset.invert
        self.query_one("#match-select", Select).value = preset.match_mode.value
        self.query_one("#clahe-switch", Switch).value = preset.clahe
        self._schedule_refresh()

    def _save_export(self, fmt: str) -> None:
        if self._image is None or self.image_path is None:
            return
        art = convert(self._image, self._options)
        stem = Path(self.image_path).stem

        if fmt == "txt":
            path = f"{stem}_ascii.txt"
            Path(path).write_text(render_text(art))
        elif fmt == "html":
            path = f"{stem}_ascii.html"
            Path(path).write_text(render_html(art))
        elif fmt == "png":
            path = f"{stem}_ascii.png"
            Path(path).write_bytes(render_png(art))
        else:
            return

        self.query_one("#status-bar", Static).update(f"Saved to {path}")
