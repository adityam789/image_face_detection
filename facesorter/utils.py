"""
facesorter/utils.py
Utility functions for face sorter pipeline.
"""
import os
import shutil
from typing import List
from PIL import Image, UnidentifiedImageError
import pillow_heif

pillow_heif.register_heif_opener()

def is_image_file(filename: str) -> bool:
    SUPPORTED_FORMATS = ('.jpg', '.jpeg', '.png', '.heic', '.JPG', '.JPEG', '.PNG', '.HEIC')
    return filename.lower().endswith(SUPPORTED_FORMATS)

def load_image(image_path: str) -> Image.Image:
    try:
        img = Image.open(image_path)
        img = img.convert('RGB')
        return img
    except UnidentifiedImageError:
        raise
    except Exception as e:
        raise

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def copy_image_to_person_folder(src_path: str, dest_dir: str, person_id: str, filename: str) -> str:
    person_folder = os.path.join(dest_dir, person_id)
    ensure_dir(person_folder)
    dest_path = os.path.join(person_folder, filename)
    if not os.path.exists(dest_path):
        shutil.copy2(src_path, dest_path)
    return dest_path

def get_image_files(source_dir: str) -> List[str]:
    files = []
    for root, _, filenames in os.walk(source_dir):
        for fname in filenames:
            if is_image_file(fname):
                files.append(os.path.join(root, fname))
    return files
