"""
test_face_detection.py
Test script for face detection and embedding using insightface with detailed logging and adjustable parameters.
"""
import argparse
import os
from PIL import Image, ExifTags, ImageDraw, ImageFont
from facesorter.detection import get_face_model, detect_faces_and_embeddings
import pillow_heif

pillow_heif.register_heif_opener()

def draw_faces_with_labels(img: Image.Image, faces):
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("Arial.ttf", 18)
    except:
        font = ImageFont.load_default()

    for idx, face in enumerate(faces):
        x1, y1, x2, y2 = face["box"]
        score = face["score"]

        label = f"Face {idx} | {score:.2f}"

        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline="red", width=4)

        # Compute text size (Pillow >=10)
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        # Label background
        label_bg = [x1, y1 - text_h - 6, x1 + text_w + 6, y1]
        draw.rectangle(label_bg, fill="red")

        # Draw text
        draw.text((x1 + 3, y1 - text_h - 3), label, fill="white", font=font)

    return img

def correct_exif_orientation(img: Image.Image) -> Image.Image:
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = img._getexif()
        if exif is not None:
            orientation = exif.get(orientation, None)
            if orientation == 3:
                img = img.rotate(180, expand=True)
            elif orientation == 6:
                img = img.rotate(270, expand=True)
            elif orientation == 8:
                img = img.rotate(90, expand=True)
    except Exception:
        pass
    return img

def test_image(image_path: str, model, confidence: float):
    print(f"\nTesting image: {image_path}")
    try:
        img = Image.open(image_path)
        img = correct_exif_orientation(img)
        img = img.convert('RGB')
        print(f"Loaded image size: {img.size}, mode: {img.mode}")
    except Exception as e:
        print(f"Failed to load image: {e}")
        return
    faces = detect_faces_and_embeddings(model, img)
    print(f"Total faces detected (before threshold): {len(faces)}")
    faces = [f for f in faces if f['score'] >= confidence]
    print(f"Faces after confidence filter (>{confidence}): {len(faces)}")
    for idx, face in enumerate(faces):
        print(f"  Face {idx}: Box={face['box']}, Score={face['score']:.3f}")
    if not faces:
        print("No faces detected above threshold.")

    # 🔥 DRAW OVERLAY
    vis_img = img.copy()
    vis_img = draw_faces_with_labels(vis_img, faces)

    vis_img.show()
    vis_img.save("debug_faces_labeled_2.jpg")

def main():
    parser = argparse.ArgumentParser(description="Test face detection with insightface and log details.")
    parser.add_argument('--image', type=str, required=True, help="Path to image file to test.")
    parser.add_argument('--confidence', type=float, default=0.5, help="Detection confidence threshold.")
    args = parser.parse_args()
    print(f"Loading face model...")
    model = get_face_model()
    test_image(args.image, model, args.confidence)

if __name__ == "__main__":
    ## main()
    print(f"Loading face model...")
    model = get_face_model()
    test_image("/Users/aditya/Documents/adiya_maheshwari/images/IMG_8474.HEIC", model, 0.7)

