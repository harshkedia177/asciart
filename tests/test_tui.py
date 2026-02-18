from asciart.tui import AsciiArtApp, FilePickerScreen


def test_tui_app_creates():
    """Smoke test: the app can be instantiated."""
    app = AsciiArtApp(image_path="nonexistent.png")
    assert app is not None


def test_tui_app_creates_without_path():
    """App can be created with no image path (will show file picker)."""
    app = AsciiArtApp()
    assert app is not None
    assert app.image_path is None


def test_file_picker_screen_creates():
    """File picker screen can be instantiated."""
    screen = FilePickerScreen()
    assert screen is not None
