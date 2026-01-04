
"""
face_sorter.py
Main entry point for the face sorting pipeline.
"""
import argparse
import os
import sys
import logging
import configparser
from facesorter.pipeline import process_images
from facesorter.finder import find_matches
from facesorter.utils import ensure_dir

def setup_logging(log_path: str) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode='w', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ],
        force=True # Reset handlers if already configured
    )

def load_config(config_path: str):
    config = configparser.ConfigParser()
    config.read(config_path)
    return config['DEFAULT'] if 'DEFAULT' in config else {}

def main():
    parser = argparse.ArgumentParser(description="Face clustering and sorting pipeline")
    subparsers = parser.add_subparsers(dest='command', required=True, help="Command to run")

    # SORT command
    parser_sort = subparsers.add_parser('sort', help="Sort directory of images into person folders")
    parser_sort.add_argument('--source', type=str, required=True, help="Source directory of images")
    parser_sort.add_argument('--output', type=str, required=True, help="Output directory for sorted images")
    parser_sort.add_argument('--config', type=str, default='config.ini', help="Path to config file")
    parser_sort.add_argument('--threshold', type=float, help="Clustering threshold (cosine distance)")
    parser_sort.add_argument('--confidence', type=float, help="Face detection confidence threshold")
    parser_sort.add_argument('--nms', type=float, help="Face detection NMS threshold")

    # FIND command
    parser_find = subparsers.add_parser('find', help="Find images of a specific person")
    parser_find.add_argument('--source', type=str, required=True, help="Source directory (search haystack)")
    parser_find.add_argument('--reference', type=str, required=True, help="Reference directory (needle faces)")
    parser_find.add_argument('--output', type=str, required=True, help="Output directory for found images")
    parser_find.add_argument('--config', type=str, default='config.ini', help="Path to config file")
    parser_find.add_argument('--threshold', type=float, help="Match threshold (cosine distance)")
    parser_find.add_argument('--confidence', type=float, help="Face detection confidence threshold")

    args = parser.parse_args()

    ensure_dir(args.output)
    log_path = os.path.join(args.output, "processing_log.txt")
    setup_logging(log_path)
    
    config = load_config(args.config)

    if args.command == 'sort':
        cluster_threshold = args.threshold if args.threshold is not None else float(config.get('cluster_threshold', 0.6))
        confidence = args.confidence if args.confidence is not None else float(config.get('confidence', 0.8))
        nms_threshold = args.nms if args.nms is not None else float(config.get('nms_threshold', 0.4))
        
        logging.info("Starting face sorting pipeline...")
        process_images(
            source_dir=args.source,
            output_dir=args.output,
            cluster_threshold=cluster_threshold,
            confidence=confidence,
            nms_threshold=nms_threshold
        )
    
    elif args.command == 'find':
        match_threshold = args.threshold if args.threshold is not None else float(config.get('cluster_threshold', 0.6)) # Reuse cluster threshold default
        confidence = args.confidence if args.confidence is not None else float(config.get('confidence', 0.8))
        
        logging.info("Starting face finder...")
        find_matches(
            source_dir=args.source,
            output_dir=args.output,
            reference_dir=args.reference,
            threshold=match_threshold,
            confidence=confidence
        )

    logging.info("Processing complete. See processing_log.txt for details.")

if __name__ == "__main__":
    main()
