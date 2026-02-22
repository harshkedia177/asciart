from __future__ import annotations

import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Optional

import typer
from PIL import Image

from asciart.models import ConvertOptions, Mode, ColorMode, DitherMode, MatchMode, OutputFormat
from asciart.presets import get_preset
from asciart.core.engine import convert
from asciart.renderers.terminal import render_ansi
from asciart.renderers.text import render_text
from asciart.renderers.html import render_html
from asciart.renderers.svg import render_svg
from asciart.renderers.image import render_png

app = typer.Typer(name="asciart", help="Convert images to stunning ASCII art.")


@app.command(name="convert")
def convert_cmd(
    image_path: Path = typer.Argument(..., help="Path to the image file", exists=True),
    width: int = typer.Option(80, "-w", "--width", help="Output width in characters"),
    mode: Mode = typer.Option(Mode.ASCII, "--mode", help="Rendering mode"),
    color: ColorMode = typer.Option(ColorMode.AUTO, "--color", help="Color mode"),
    dither: DitherMode = typer.Option(DitherMode.NONE, "--dither", help="Dithering algorithm"),
    chars: str = typer.Option(
        "standard", "--chars",
        help="Character ramp: standard, detailed, minimal, alphabets, numbers, alphanumeric, or a custom string",
    ),
    invert: bool = typer.Option(False, "-i", "--invert", help="Invert brightness"),
    brightness: float = typer.Option(0.0, "--brightness", help="Brightness offset (-100 to 100)"),
    contrast: float = typer.Option(1.0, "--contrast", help="Contrast multiplier"),
    font_ratio: float = typer.Option(0.5, "--ratio", help="Font cell width/height ratio"),
    preset: Optional[str] = typer.Option(None, "--preset", help="Use a preset"),
    animate: bool = typer.Option(False, "--animate", help="Animate GIF in terminal"),
    format: OutputFormat = typer.Option(OutputFormat.ANSI, "--format", help="Output format"),
    output: Optional[Path] = typer.Option(None, "-o", "--output", help="Save to file"),
    copy: bool = typer.Option(False, "--copy", help="Copy output to clipboard"),
    match: MatchMode = typer.Option(MatchMode.BRIGHTNESS, "--match", help="Character matching mode"),
    font_path: Optional[str] = typer.Option(None, "--font", help="Font path for structural/hybrid matching"),
    clahe: bool = typer.Option(False, "--clahe", help="Enable CLAHE local contrast enhancement"),
    backend_name: Optional[str] = typer.Option(None, "--backend", help="Force backend: numpy, numba, cupy"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show backend and timing info"),
):
    """Convert an image to ASCII art."""
    if backend_name:
        from asciart.accel import force_backend
        try:
            force_backend(backend_name)
        except ValueError as e:
            typer.echo(str(e), err=True)
            raise typer.Exit(code=1)

    if verbose:
        from asciart.accel import get_backend
        backend = get_backend()
        typer.echo(f"Backend: {backend.name} (GPU={backend.has_gpu}, JIT={backend.has_jit})", err=True)

    img = Image.open(image_path)

    if preset:
        base = get_preset(preset)
        if base is None:
            typer.echo(f"Unknown preset '{preset}'. Available: photo, logo, retro, hd, blocks, lineart, studio", err=True)
            raise typer.Exit(code=1)
        options = replace(base, width=width, font_ratio=font_ratio)
        if invert:
            options = replace(options, invert=True)
    else:
        options = ConvertOptions(
            width=width,
            mode=mode,
            chars=chars,
            color=color,
            dither=dither,
            invert=invert,
            brightness=brightness,
            contrast=contrast,
            font_ratio=font_ratio,
            match_mode=match,
            font_path=font_path,
            clahe=clahe,
        )

    if animate and str(image_path).lower().endswith(".gif"):
        from asciart.core.gif import extract_frames
        frames_pil, durations = extract_frames(str(image_path))
        arts = [convert(f, options) for f in frames_pil]

        sys.stdout.write("\033[?25l")  # hide cursor
        try:
            while True:
                for art_frame, duration in zip(arts, durations):
                    if format == OutputFormat.ANSI:
                        rendered = render_ansi(art_frame, use_256=(color == ColorMode.ANSI256))
                    else:
                        rendered = render_text(art_frame)
                    sys.stdout.write("\033[H")
                    sys.stdout.write(rendered)
                    sys.stdout.flush()
                    time.sleep(duration / 1000.0)
        except KeyboardInterrupt:
            pass
        finally:
            sys.stdout.write("\033[?25h")  # show cursor
            sys.stdout.write("\n")
        return

    if verbose:
        t0 = time.perf_counter()
    art = convert(img, options)
    if verbose:
        elapsed = (time.perf_counter() - t0) * 1000
        typer.echo(f"Conversion: {elapsed:.0f}ms ({art.width}x{art.height} chars)", err=True)

    if format == OutputFormat.TEXT:
        rendered = render_text(art)
    elif format == OutputFormat.ANSI:
        use_256 = color == ColorMode.ANSI256
        rendered = render_ansi(art, use_256=use_256)
    elif format == OutputFormat.HTML:
        rendered = render_html(art)
    elif format == OutputFormat.SVG:
        rendered = render_svg(art)
    elif format == OutputFormat.PNG:
        if not output:
            typer.echo("PNG format requires --output/-o flag.", err=True)
            raise typer.Exit(code=1)
        data = render_png(art)
        output.write_bytes(data)
        typer.echo(f"Saved PNG to {output}")
        return

    if output:
        output.write_text(rendered)
        typer.echo(f"Saved to {output}")
    else:
        typer.echo(rendered)

    if copy:
        from asciart.clipboard import copy_to_clipboard
        plain = render_text(art)
        if copy_to_clipboard(plain):
            typer.echo("Copied to clipboard!", err=True)
        else:
            typer.echo("Failed to copy to clipboard (is xclip/pbcopy installed?)", err=True)


@app.command()
def play(
    image_path: Optional[Path] = typer.Argument(None, help="Path to the image file"),
):
    """Open interactive mode with live preview. Omit image path to browse for a file."""
    from asciart.tui import AsciiArtApp
    path = str(image_path) if image_path else None
    tui = AsciiArtApp(image_path=path)
    tui.run()


if __name__ == "__main__":
    app()
