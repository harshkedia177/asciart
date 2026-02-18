import numpy as np

from asciart.core.contrast import apply_clahe


class TestClaheShape:
    def test_clahe_returns_same_shape(self):
        """CLAHE output must have the same shape as the input."""
        gray = np.random.uniform(0, 255, (64, 80))
        result = apply_clahe(gray)
        assert result.shape == gray.shape

    def test_clahe_returns_same_shape_non_square(self):
        gray = np.random.uniform(0, 255, (100, 200))
        result = apply_clahe(gray)
        assert result.shape == gray.shape

    def test_clahe_returns_same_shape_small(self):
        gray = np.random.uniform(0, 255, (8, 8))
        result = apply_clahe(gray)
        assert result.shape == gray.shape


class TestClaheRange:
    def test_clahe_output_in_valid_range(self):
        """All output values must be between 0 and 255."""
        gray = np.random.uniform(0, 255, (64, 64))
        result = apply_clahe(gray)
        assert result.min() >= 0.0
        assert result.max() <= 255.0

    def test_clahe_output_range_with_extreme_values(self):
        """Edge case: input that includes exact 0 and 255 values."""
        gray = np.zeros((32, 32))
        gray[0, 0] = 0.0
        gray[-1, -1] = 255.0
        result = apply_clahe(gray)
        assert result.min() >= 0.0
        assert result.max() <= 255.0


class TestClaheDynamicRange:
    def test_clahe_improves_dynamic_range(self):
        """A low-contrast input should produce a wider range output."""
        # Create a low-contrast image: all values between 100-110
        gray = np.random.uniform(100, 110, (64, 64))
        result = apply_clahe(gray)

        input_range = gray.max() - gray.min()
        output_range = result.max() - result.min()
        assert output_range > input_range, (
            f"Expected output range ({output_range:.1f}) > input range ({input_range:.1f})"
        )


class TestClaheUniform:
    def test_clahe_uniform_image_roughly_unchanged(self):
        """A perfectly uniform image should remain roughly uniform after CLAHE."""
        gray = np.full((64, 64), 128.0)
        result = apply_clahe(gray)

        # All pixels had the same value, so the output should be very uniform
        # (it maps to a single CDF value).  Allow small float tolerance.
        assert result.max() - result.min() < 1.0, (
            f"Uniform input produced spread of {result.max() - result.min():.2f}"
        )
