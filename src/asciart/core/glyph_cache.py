from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.signal import convolve2d


@dataclass
class GlyphEntry:
    """Pre-rendered glyph with extracted structural features."""

    char: str
    index: int
    bitmap: np.ndarray  # (cell_h, cell_w) float [0, 1]
    mean_brightness: float
    features: np.ndarray  # (grid_h * grid_w,) float -- sub-block means
    variance: float
    frequency: float


class GlyphMatcher:
    """Pre-renders candidate characters to bitmaps and matches image tiles
    by structural/hybrid similarity.

    Supported match modes:
      - brightness: compare mean brightness only
      - structural: sum-of-squared-differences on sub-block feature vectors
      - hybrid: weighted combination of brightness, structure, variance, frequency
    """

    def __init__(
        self,
        chars: str,
        cell_w: int = 6,
        cell_h: int = 10,
        font_path: str | None = None,
        grid: tuple[int, int] = (4, 4),
    ) -> None:
        self.chars = chars
        self.cell_w = cell_w
        self.cell_h = cell_h
        self.grid = grid

        font = self._load_font(font_path, cell_h)

        self.glyphs: list[GlyphEntry] = []
        for i, ch in enumerate(chars):
            bitmap = self._render_char(ch, font)
            features = self._extract_features(bitmap)
            mean_brightness = float(bitmap.mean())
            variance = float(bitmap.var())
            frequency = self._spatial_frequency(bitmap)
            self.glyphs.append(
                GlyphEntry(
                    char=ch,
                    index=i,
                    bitmap=bitmap,
                    mean_brightness=mean_brightness,
                    features=features,
                    variance=variance,
                    frequency=frequency,
                )
            )

        # Adaptive hybrid weights: lerp from brightness-heavy (n<=4) to original (n>=8)
        t = np.clip((len(chars) - 4) / 4.0, 0.0, 1.0)
        self.w_brightness = 0.70 * (1.0 - t) + 0.25 * t
        self.w_structure = 0.20 * (1.0 - t) + 0.50 * t
        self.w_variance = 0.05 * (1.0 - t) + 0.15 * t
        self.w_frequency = 0.05 * (1.0 - t) + 0.10 * t

        # Pre-compute matrices for batch operations
        self.brightness_array = np.array(
            [g.mean_brightness for g in self.glyphs], dtype=np.float64
        )
        self.feature_matrix = np.array(
            [g.features for g in self.glyphs], dtype=np.float64
        )  # (N, grid_h * grid_w)
        self.variance_array = np.array(
            [g.variance for g in self.glyphs], dtype=np.float64
        )
        self.frequency_array = np.array(
            [g.frequency for g in self.glyphs], dtype=np.float64
        )

    @staticmethod
    def _load_font(font_path: str | None, cell_h: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        """Load a font, with fallback to default."""
        if font_path is not None:
            try:
                return ImageFont.truetype(font_path, size=cell_h)
            except (OSError, IOError):
                pass
        # Try common monospace fonts
        for name in ("Courier", "Courier New", "DejaVu Sans Mono", "Liberation Mono"):
            try:
                return ImageFont.truetype(name, size=cell_h)
            except (OSError, IOError):
                continue
        return ImageFont.load_default()

    def _render_char(self, char: str, font: ImageFont.FreeTypeFont | ImageFont.ImageFont) -> np.ndarray:
        """Render a single character to a (cell_h, cell_w) float [0, 1] bitmap."""
        img = Image.new("L", (self.cell_w, self.cell_h), 0)
        draw = ImageDraw.Draw(img)
        draw.text((0, 0), char, fill=255, font=font)
        bitmap = np.array(img, dtype=np.float64) / 255.0
        return bitmap

    def _extract_features(self, bitmap: np.ndarray) -> np.ndarray:
        """Extract grid_h x grid_w sub-block mean brightness features."""
        grid_h, grid_w = self.grid
        h, w = bitmap.shape

        # Compute sub-block boundaries (handles non-divisible sizes)
        row_splits = np.linspace(0, h, grid_h + 1, dtype=int)
        col_splits = np.linspace(0, w, grid_w + 1, dtype=int)

        features = np.zeros(grid_h * grid_w, dtype=np.float64)
        idx = 0
        for r in range(grid_h):
            for c in range(grid_w):
                block = bitmap[row_splits[r] : row_splits[r + 1], col_splits[c] : col_splits[c + 1]]
                features[idx] = block.mean() if block.size > 0 else 0.0
                idx += 1
        return features

    @staticmethod
    def _spatial_frequency(bitmap: np.ndarray) -> float:
        """Compute spatial frequency via Laplacian filter energy."""
        laplacian_kernel = np.array(
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64
        )
        if bitmap.shape[0] < 3 or bitmap.shape[1] < 3:
            return 0.0
        response = convolve2d(bitmap, laplacian_kernel, mode="same", boundary="fill")
        return float(np.mean(response ** 2))

    @staticmethod
    def _normalize(arr: np.ndarray) -> np.ndarray:
        """Normalize array to [0, 1] range. Returns unchanged if max is 0."""
        m = arr.max()
        return arr / m if m > 0 else arr

    def match_tile(self, tile: np.ndarray, mode: str = "structural") -> int:
        """Match a single (cell_h, cell_w) tile to the best glyph index.

        Args:
            tile: (cell_h, cell_w) float [0, 1] grayscale tile.
            mode: "brightness", "structural", or "hybrid".

        Returns:
            Index into self.chars of the best-matching glyph.
        """
        tile_mean = float(tile.mean())

        if mode == "brightness":
            diffs = np.abs(self.brightness_array - tile_mean)
            return int(np.argmin(diffs))

        tile_features = self._extract_features(tile)

        if mode == "structural":
            ssd = np.sum((self.feature_matrix - tile_features) ** 2, axis=1)
            return int(np.argmin(ssd))

        # hybrid
        combined = self._hybrid_score(
            tile_means=tile_mean,
            tile_features=tile_features,
            tile_variances=float(tile.var()),
            tile_freqs=self._spatial_frequency(tile),
        )
        return int(np.argmin(combined))

    def _hybrid_score(
        self,
        tile_means: float | np.ndarray,
        tile_features: np.ndarray,
        tile_variances: float | np.ndarray,
        tile_freqs: float | np.ndarray,
    ) -> np.ndarray:
        """Compute weighted hybrid similarity scores.

        Works for both single tiles (scalars) and batches (arrays with newaxis).
        Returns shape (N_glyphs,) for single tile or (n_tiles, N_glyphs) for batch.
        """
        norm = self._normalize

        if isinstance(tile_means, np.ndarray):
            brightness_diff = np.abs(
                tile_means[:, np.newaxis] - self.brightness_array[np.newaxis, :]
            )
            ssd = np.sum(
                (tile_features[:, np.newaxis, :] - self.feature_matrix[np.newaxis, :, :]) ** 2,
                axis=2,
            )
            variance_diff = np.abs(
                tile_variances[:, np.newaxis] - self.variance_array[np.newaxis, :]
            )
            frequency_diff = np.abs(
                tile_freqs[:, np.newaxis] - self.frequency_array[np.newaxis, :]
            )
        else:
            brightness_diff = np.abs(self.brightness_array - tile_means)
            ssd = np.sum((self.feature_matrix - tile_features) ** 2, axis=1)
            variance_diff = np.abs(self.variance_array - tile_variances)
            frequency_diff = np.abs(self.frequency_array - tile_freqs)

        return (
            self.w_brightness * norm(brightness_diff)
            + self.w_structure * norm(ssd)
            + self.w_variance * norm(variance_diff)
            + self.w_frequency * norm(frequency_diff)
        )

    def match_tiles_batch(
        self, tiles: np.ndarray, mode: str = "structural"
    ) -> np.ndarray:
        """Match a grid of tiles to glyph indices.

        Args:
            tiles: (grid_h, grid_w, cell_h, cell_w) float [0, 1] grayscale tiles.
            mode: "brightness", "structural", or "hybrid".

        Returns:
            (grid_h, grid_w) uint8 array of glyph indices.
        """
        gh, gw = tiles.shape[:2]

        if mode == "brightness":
            tile_means = tiles.mean(axis=(2, 3)).reshape(-1)
            diffs = np.abs(tile_means[:, np.newaxis] - self.brightness_array[np.newaxis, :])
            flat_indices = np.argmin(diffs, axis=1).astype(np.uint8)
            return flat_indices.reshape(gh, gw)

        tile_features = self._extract_features_batch(tiles)

        if mode == "structural":
            ssd = np.sum(
                (tile_features[:, np.newaxis, :] - self.feature_matrix[np.newaxis, :, :]) ** 2,
                axis=2,
            )
            flat_indices = np.argmin(ssd, axis=1).astype(np.uint8)
            return flat_indices.reshape(gh, gw)

        # hybrid
        n_tiles = gh * gw
        flat_tiles = tiles.reshape(n_tiles, tiles.shape[2], tiles.shape[3])
        combined = self._hybrid_score(
            tile_means=tiles.mean(axis=(2, 3)).reshape(-1),
            tile_features=tile_features,
            tile_variances=tiles.var(axis=(2, 3)).reshape(-1),
            tile_freqs=np.array(
                [self._spatial_frequency(flat_tiles[i]) for i in range(n_tiles)],
                dtype=np.float64,
            ),
        )
        flat_indices = np.argmin(combined, axis=1).astype(np.uint8)
        return flat_indices.reshape(gh, gw)

    def _extract_features_batch(self, tiles: np.ndarray) -> np.ndarray:
        """Extract features for a batch of tiles.

        Args:
            tiles: (grid_h, grid_w, cell_h, cell_w) float array.

        Returns:
            (grid_h * grid_w, n_features) float array.
        """
        gh, gw, ch, cw = tiles.shape
        n_tiles = gh * gw
        flat_tiles = tiles.reshape(n_tiles, ch, cw)

        grid_h, grid_w = self.grid
        row_splits = np.linspace(0, ch, grid_h + 1, dtype=int)
        col_splits = np.linspace(0, cw, grid_w + 1, dtype=int)

        n_features = grid_h * grid_w
        features = np.zeros((n_tiles, n_features), dtype=np.float64)

        idx = 0
        for r in range(grid_h):
            for c in range(grid_w):
                block = flat_tiles[
                    :, row_splits[r] : row_splits[r + 1], col_splits[c] : col_splits[c + 1]
                ]
                features[:, idx] = block.mean(axis=(1, 2)) if block.size > 0 else 0.0
                idx += 1
        return features
