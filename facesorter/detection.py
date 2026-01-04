"""
facesorter/detection.py
Face detection and embedding extraction using insightface (RetinaFace + ArcFace).
"""
from typing import List, Dict, Any
from PIL import Image
import insightface
import numpy as np

def get_face_model() -> insightface.app.FaceAnalysis:
    """Initialize and return the InsightFace analysis model."""
    model = insightface.app.FaceAnalysis(providers=['CPUExecutionProvider'])
    model.prepare(ctx_id=0)
    return model

def detect_faces_and_embeddings(model: insightface.app.FaceAnalysis, image: Image.Image) -> List[Dict[str, Any]]:
    """
    Detect faces in an image and extract embeddings.
    
    Returns:
        List of dicts containing 'box', 'embedding', 'score', 'face_idx'.
    """
    # insightface expects numpy array in BGR
    img_np = np.array(image)[:, :, ::-1]
    faces = model.get(img_np)
    results = []
    for idx, face in enumerate(faces):
        box = face.bbox.astype(int).tolist()  # [x1, y1, x2, y2]
        embedding = face.embedding
        results.append({
            'box': box,
            'embedding': embedding,
            'score': face.det_score,
            'face_idx': idx
        })
    return results

def extract_face(img: Image.Image, box: List[int]) -> Image.Image:
    """Crop face from image using bounding box."""
    x1, y1, x2, y2 = box
    return img.crop((x1, y1, x2, y2))
