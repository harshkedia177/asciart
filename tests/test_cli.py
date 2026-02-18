from typer.testing import CliRunner
from asciart.cli import app
from PIL import Image
import tempfile
from pathlib import Path

runner = CliRunner()


def _create_test_image(path: Path):
    img = Image.new("RGB", (100, 100), (128, 128, 128))
    img.save(str(path))


def test_convert_basic():
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        _create_test_image(Path(f.name))
        result = runner.invoke(app, ["convert", f.name, "-w", "20", "--color", "none"])
        assert result.exit_code == 0
        lines = result.output.strip().split("\n")
        assert len(lines) > 0
        assert len(lines[0]) == 20


def test_convert_save_to_file():
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as img_f:
        _create_test_image(Path(img_f.name))
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as out_f:
            result = runner.invoke(
                app, ["convert", img_f.name, "-w", "20", "--format", "text", "-o", out_f.name]
            )
            assert result.exit_code == 0
            content = Path(out_f.name).read_text()
            assert len(content) > 0
