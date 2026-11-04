"""Draw detected face boxes and confidence scores on an image.

A small debugging aid for checking what the detector sees, e.g.::

    python scripts/visualize_detections.py --image photo.jpg --confidence 0.6 --output out.jpg

Requires the package to be importable: run ``pip install -e .`` first (or use
``PYTHONPATH=.`` from the repository root). The output image contains the faces in the
photo, so keep it out of version control.
"""

from __future__ import annotations

import argparse
import sys

from PIL import Image, ImageDraw, ImageFont

from facesorter.detection import detect_faces_and_embeddings, get_face_model
from facesorter.utils import load_image


def draw_faces_with_labels(img: Image.Image, faces) -> Image.Image:
    """Return a copy of ``img`` with a box and ``Face N | score`` label drawn per face."""
    img = img.copy()
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("Arial.ttf", 18)
    except OSError:
        font = ImageFont.load_default()

    for idx, face in enumerate(faces):
        x1, y1, x2, y2 = face.box
        label = f"Face {idx} | {face.score:.2f}"

        draw.rectangle([x1, y1, x2, y2], outline="red", width=4)

        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        text_w, text_h = right - left, bottom - top
        draw.rectangle([x1, y1 - text_h - 6, x1 + text_w + 6, y1], fill="red")
        draw.text((x1 + 3, y1 - text_h - 3), label, fill="white", font=font)

    return img


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize face detections on a single image.")
    parser.add_argument("--image", required=True, help="Path to the image to analyse.")
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Minimum detection score to draw (default: 0.5).",
    )
    parser.add_argument(
        "--output",
        default="detections.jpg",
        help="Where to save the annotated image (default: detections.jpg).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Also open the annotated image in the system viewer.",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    try:
        img = load_image(args.image)
    except Exception as exc:
        print(f"Failed to load {args.image}: {exc}", file=sys.stderr)
        return 1
    print(f"Loaded {args.image}: size={img.size}, mode={img.mode}")

    print("Loading face model...")
    model = get_face_model()
    detections = detect_faces_and_embeddings(model, img)
    print(f"Faces detected (before threshold): {len(detections)}")

    faces = [f for f in detections if f.score >= args.confidence]
    print(f"Faces with score >= {args.confidence}: {len(faces)}")
    for idx, face in enumerate(faces):
        print(f"  Face {idx}: box={face.box}, score={face.score:.3f}")

    annotated = draw_faces_with_labels(img, faces)
    annotated.save(args.output)
    print(f"Saved annotated image to {args.output}")
    if args.show:
        annotated.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
