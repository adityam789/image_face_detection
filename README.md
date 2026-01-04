# Face Sorter & Finder

A powerful Python tool to organize your photo collection using advanced face recognition. Detects, clusters, and sorts photos by person, or finds all photos of a specific person from a large dataset.

## Features

- **Face Sorting**: Automatically group photos of the same person together.
- **Face Finding**: Find all photos of a specific person using a reference image.
- **Robust Detection**: Powered by `insightface` (RetinaFace + ArcFace) for state-of-the-art accuracy.
- **HEIC Support**: Native support for iPhone .HEIC photos.
- **Customizable**: Adjustable confidence and clustering thresholds via CLI or config.

## Installation

### Prerequisites
- Python 3.8+
- Recommended: A virtual environment

### Steps
1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd image_face_detection
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

The tool provides two main commands: `sort` and `find`.

### 1. Sorting Photos
Organize a chaotic folder of images into named folders (e.g., `Person_0`, `Person_1`, ...).

```bash
python face_sorter.py sort --source ./my_photos --output ./sorted_results
```

**Options:**
- `--threshold`: Clustering distance threshold (default: 0.6). Lower = stricter (fewer false positives, more splits). Higher = looser.
- `--confidence`: Minimum face detection score (default: 0.8).

### 2. Finding a Person
Find all photos of **one specific person** from a large collection.

```bash
python face_sorter.py find \
    --reference ./reference_photos/bob \
    --source ./all_photos \
    --output ./found_bob
```

- `--reference`: Directory containing one or more clear photos of the target person.
- `--output`: Where the matching photos will be copied.
- `--threshold`: Match distance threshold (default: 0.6).

## Configuration
You can also set default parameters in `config.ini`:
```ini
[DEFAULT]
confidence = 0.8
cluster_threshold = 0.6
```

## Troubleshooting
- **No faces found?** Lower the `--confidence` (e.g., to 0.5) to detect smaller or blurrier faces.
- **Mixed people in one folder?** Lower the `--threshold` (e.g., to 0.4) to make the clustering stricter.
- **Same person split into multiple folders?** Increase the `--threshold` (e.g., to 0.7).

## License
MIT
