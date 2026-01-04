"""
facesorter/finder.py
Logic for finding a specific face in a directory of images.
"""
import logging
import os
import shutil
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from .utils import get_image_files, load_image, ensure_dir
from .detection import detect_faces_and_embeddings, get_face_model
from typing import List, Optional

def get_reference_embedding(model, reference_dir: str, confidence: float = 0.5) -> Optional[np.ndarray]:
    """
    Computes the average embedding for a person given a directory of their photos.
    """
    files = get_image_files(reference_dir)
    if not files:
        logging.error(f"No image files found in reference directory: {reference_dir}")
        return None

    embeddings = []
    
    logging.info(f"Computing reference embedding from {len(files)} images...")
    
    for file_path in tqdm(files, desc="Analyzing reference images"):
        try:
            img = load_image(file_path)
            faces = detect_faces_and_embeddings(model, img)
            # Filter by confidence
            valid_faces = [f for f in faces if f['score'] >= confidence]
            
            if not valid_faces:
                logging.warning(f"No valid faces found in {file_path}")
                continue
                
            if len(valid_faces) > 1:
                # Heuristic: Take the largest face (probably the subject)
                # Box is [x1, y1, x2, y2], area = (x2-x1)*(y2-y1)
                valid_faces.sort(key=lambda f: (f['box'][2]-f['box'][0]) * (f['box'][3]-f['box'][1]), reverse=True)
                logging.debug(f"Multiple faces in {file_path}, using largest one.")
            
            embeddings.append(valid_faces[0]['embedding'])
            
        except Exception as e:
            logging.warning(f"Error processing matching image {file_path}: {e}")

    if not embeddings:
        logging.error("Could not extract any valid face embeddings from reference images.")
        return None
        
    # Compute centroid (average embedding)
    embeddings_np = np.array(embeddings)
    centroid = np.mean(embeddings_np, axis=0)
    
    # Normalize the centroid (important for cosine similarity)
    norm = np.linalg.norm(centroid)
    if norm > 0:
        centroid = centroid / norm
        
    logging.info(f"Reference embedding computed from {len(embeddings)} faces.")
    return centroid

def find_matches(source_dir: str, output_dir: str, reference_dir: str, threshold: float = 0.5, confidence: float = 0.5):
    """
    Finds matches of the reference person in the source directory.
    """
    # 1. Get Model
    model = get_face_model()
    
    # 2. Get Reference Embedding
    ref_embedding = get_reference_embedding(model, reference_dir, confidence)
    if ref_embedding is None:
        return

    # 3. Scan Source Directory
    image_files = get_image_files(source_dir)
    logging.info(f"Scanning {len(image_files)} images in {source_dir}...")
    
    ensure_dir(output_dir)
    matches_found = 0
    
    # Pre-shape for cosine_similarity (1, N)
    ref_embedding = ref_embedding.reshape(1, -1)
    
    for img_path in tqdm(image_files, desc="Searching for matches"):
        try:
            img = load_image(img_path)
            faces = detect_faces_and_embeddings(model, img)
            
            # Check each face in the image
            is_match = False
            for face in faces:
                if face['score'] < confidence:
                    continue
                
                emb = face['embedding'].reshape(1, -1)
                # Calculate similarity (higher is better, range [-1, 1])
                # We often use distance = 1 - sim in clustering, but here we just want high sim.
                # Threshold passed as args.threshold is usually "distance threshold" from the previous CLI?
                # Actually, in clustering DBSCAN uses distance (lower is closer).
                # But typically for "find this face", people think 'sensitivity'.
                # Let's stick to Distance Threshold to be consistent with clustering logic.
                # Distance = 1 - CosineSimilarity. Distance < Threshold => Match.
                
                sim = cosine_similarity(ref_embedding, emb)[0][0]
                dist = 1.0 - sim
                
                if dist < threshold:
                    is_match = True
                    break # Found the person in this photo
            
            if is_match:
                matches_found += 1
                fname = os.path.basename(img_path)
                dest = os.path.join(output_dir, fname)
                shutil.copy2(img_path, dest)
                
        except Exception as e:
            logging.error(f"Error processing {img_path}: {e}")
            
    logging.info(f"Search complete. Found {matches_found} matching images copied to {output_dir}")
