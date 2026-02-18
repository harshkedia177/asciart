from unittest.mock import patch, MagicMock
from asciart.clipboard import copy_to_clipboard


def test_clipboard_calls_pbcopy_on_mac():
    with patch("asciart.clipboard.sys") as mock_sys:
        mock_sys.platform = "darwin"
        with patch("asciart.clipboard.subprocess.Popen") as mock_popen:
            mock_proc = MagicMock()
            mock_proc.returncode = 0
            mock_proc.communicate.return_value = (None, None)
            mock_popen.return_value = mock_proc

            result = copy_to_clipboard("test text")
            assert result is True
            mock_popen.assert_called_once_with(["pbcopy"], stdin=-1)


def test_clipboard_returns_false_on_file_not_found():
    with patch("asciart.clipboard.sys") as mock_sys:
        mock_sys.platform = "darwin"
        with patch("asciart.clipboard.subprocess.Popen", side_effect=FileNotFoundError):
            result = copy_to_clipboard("test")
            assert result is False
