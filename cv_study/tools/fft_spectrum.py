#!/usr/bin/env python3
"""Generate a centered 2D Fourier magnitude spectrum image."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


COLORMAPS = {
    "gray": None,
    "jet": cv2.COLORMAP_JET,
    "viridis": cv2.COLORMAP_VIRIDIS,
    "inferno": cv2.COLORMAP_INFERNO,
    "magma": cv2.COLORMAP_MAGMA,
    "plasma": cv2.COLORMAP_PLASMA,
    "turbo": cv2.COLORMAP_TURBO,
}


def read_grayscale_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def fourier_spectrum(image: np.ndarray) -> np.ndarray:
    image_float = image.astype(np.float32)
    fft = np.fft.fft2(image_float)
    fft_shifted = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shifted)
    log_magnitude = np.log1p(magnitude)

    spectrum = cv2.normalize(log_magnitude, None, 0, 255, cv2.NORM_MINMAX)
    return spectrum.astype(np.uint8)


def apply_colormap(spectrum: np.ndarray, name: str) -> np.ndarray:
    colormap = COLORMAPS[name]
    if colormap is None:
        return spectrum
    return cv2.applyColorMap(spectrum, colormap)


def default_output_path(input_path: Path) -> Path:
    script_dir = Path(__file__).resolve().parent
    return script_dir / f"{input_path.stem}_spectrum.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read an image and save its centered 2D Fourier spectrum."
    )
    parser.add_argument("image", type=Path, help="Input image path.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output path. Defaults to tools/<input_stem>_spectrum.png.",
    )
    parser.add_argument(
        "--color-map",
        choices=sorted(COLORMAPS),
        default="gray",
        help="Optional color map for the saved spectrum image.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the input image and spectrum in OpenCV windows.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Only display the result; do not write an output file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image = read_grayscale_image(args.image)
    spectrum = fourier_spectrum(image)
    output_image = apply_colormap(spectrum, args.color_map)

    if not args.no_save:
        output_path = args.output or default_output_path(args.image)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        ok = cv2.imwrite(str(output_path), output_image)
        if not ok:
            raise OSError(f"Could not write output image: {output_path}")
        print(f"Saved spectrum: {output_path}")

    if args.show:
        cv2.imshow("Input Image (Grayscale)", image)
        cv2.imshow("Centered Fourier Spectrum", output_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
