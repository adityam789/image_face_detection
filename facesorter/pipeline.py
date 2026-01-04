"""
facesorter/pipeline.py
Main pipeline logic for face sorting.
"""
import logging
from collections import defaultdict
from tqdm import tqdm
from .utils import load_image, get_image_files, copy_image_to_person_folder, ensure_dir
from .detection import get_face_model, detect_faces_and_embeddings, extract_face
from .clustering import cluster_embeddings
import os

def process_images(source_dir: str, output_dir: str, cluster_threshold: float, confidence: float, nms_threshold: float):
    image_files = get_image_files(source_dir)
    logging.info(f"Found {len(image_files)} image files in {source_dir}")
    face_model = get_face_model()
    face_records = []
    images_with_no_faces = []
    images_with_errors = []
    images_with_multiple_faces = set()
    for img_path in tqdm(image_files, desc="Processing images"):
        try:
            img = load_image(img_path)
        except Exception as e:
            logging.warning(f"Failed to load image {img_path}: {e}")
            images_with_errors.append(img_path)
            continue
        faces_old = detect_faces_and_embeddings(face_model, img)
        # Filter by confidence threshold
        faces = [f for f in faces_old if f['score'] >= confidence]
        if not faces:
            images_with_no_faces.append(img_path)
            logging.info(f"No faces detected in {img_path}")
            logging.info(f"Faces & their confidences {[(index, f['score']) for index, f in enumerate(faces_old)]}")
            continue
        if len(faces) > 1:
            images_with_multiple_faces.add(img_path)
        for face in faces:
            face_img = extract_face(img, face['box'])
            face_records.append({
                'embedding': face['embedding'],
                'image_path': img_path,
                'box': face['box'],
                'face_idx': face['face_idx']
            })
    if not face_records:
        logging.error("No faces with valid embeddings found. Exiting.")
        return
    embeddings = [rec['embedding'] for rec in face_records]
    labels = cluster_embeddings(embeddings, cluster_threshold)
    n_persons = len(set(labels))
    logging.info(f"Clustering complete. Found {n_persons} unique persons.")
    person_to_images = defaultdict(set)
    for rec, label in zip(face_records, labels):
        person_id = f"Person_{label}"
        person_to_images[person_id].add(rec['image_path'])
    copied = set()
    for person_id, img_paths in tqdm(person_to_images.items(), desc="Copying images"):
        for img_path in img_paths:
            filename = os.path.basename(img_path)
            dest_path = os.path.join(output_dir, person_id, filename)
            if dest_path in copied:
                continue
            try:
                copy_image_to_person_folder(img_path, output_dir, person_id, filename)
                copied.add(dest_path)
            except Exception as e:
                logging.warning(f"Failed to copy {img_path} to {dest_path}: {e}")
    summary = [
        f"Total images processed: {len(image_files)}",
        f"Total faces detected: {len(face_records)}",
        f"Unique persons found: {n_persons}",
        f"Images with multiple faces: {len(images_with_multiple_faces)}",
        f"Images with no faces: {len(images_with_no_faces)}",
        f"Images failed to load: {len(images_with_errors)}"
    ]
    logging.info("\n".join(summary))
    with open(os.path.join(output_dir, "processing_log.txt"), "w", encoding="utf-8") as logf:
        logf.write("Face Sorting Log\n")
        logf.write("="*40 + "\n")
        for line in summary:
            logf.write(line + "\n")
        logf.write("\nImages with no faces:\n")
        for path in images_with_no_faces:
            logf.write(path + "\n")
        logf.write("\nImages failed to load:\n")
        for path in images_with_errors:
            logf.write(path + "\n")
        logf.write("\nImages with multiple faces:\n")
        for path in images_with_multiple_faces:
            logf.write(path + "\n")
