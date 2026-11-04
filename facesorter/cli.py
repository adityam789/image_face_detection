"""
facesorter/cli.py
Command-line interface: ``facesorter sort`` and ``facesorter find``.

Heavy modules (scikit-learn, InsightFace) are imported only when a command actually
runs, so ``--help`` and ``--version`` stay fast.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from . import __version__
from .config import DEFAULT_CONFIG_PATH as DEFAULT_CONFIG
from .config import Settings, load_settings
from .errors import ConfigError, FaceSorterError, InvalidPathError
from .paths import is_inside

LOG_NAME = "processing_log.txt"

logger = logging.getLogger("facesorter")


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with ``sort`` and ``find`` subcommands."""
    # -v is accepted both before and after the subcommand; SUPPRESS keeps the
    # subparser from overwriting a value given before it.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Show debug output",
    )
    common.add_argument(
        "--config",
        default=None,
        help=f"Path to config file (default: ./{DEFAULT_CONFIG} if present)",
    )
    common.add_argument("--confidence", type=float, help="Face detection confidence threshold")

    parser = argparse.ArgumentParser(
        prog="facesorter", description="Cluster photos by person, or find photos of one person."
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show debug output")
    sub = parser.add_subparsers(dest="command", required=True, metavar="{sort,find}")

    p_sort = sub.add_parser(
        "sort", parents=[common], help="Sort a directory of images into person folders"
    )
    p_sort.add_argument("--source", required=True, help="Source directory of images")
    p_sort.add_argument("--output", required=True, help="Output directory for sorted images")
    p_sort.add_argument("--threshold", type=float, help="Clustering threshold (cosine distance)")

    p_find = sub.add_parser("find", parents=[common], help="Find images of a specific person")
    p_find.add_argument("--source", required=True, help="Source directory (search haystack)")
    p_find.add_argument("--reference", required=True, help="Directory of reference photos")
    p_find.add_argument("--output", required=True, help="Output directory for found images")
    p_find.add_argument("--threshold", type=float, help="Match threshold (cosine distance)")
    return parser


def setup_logging(log_path: str, verbose: bool = False) -> list[logging.Handler]:
    """
    Attach a file handler (``log_path``) and a stderr handler to the ``facesorter`` logger.

    Returns the handlers added; pass them to :func:`teardown_logging` when done.
    """
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    handlers: list[logging.Handler] = [
        logging.FileHandler(log_path, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stderr),
    ]
    for handler in handlers:
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.propagate = False
    return handlers


def teardown_logging(handlers: list[logging.Handler]) -> None:
    """Remove and close handlers installed by :func:`setup_logging` and reset the logger."""
    for handler in handlers:
        logger.removeHandler(handler)
        handler.close()
    logger.setLevel(logging.NOTSET)
    logger.propagate = True


def _check_paths(args: argparse.Namespace) -> None:
    """Fail early, before creating anything, if the inputs or output are unusable."""
    inputs = [("source", args.source), ("reference", getattr(args, "reference", None))]
    for label, path in inputs:
        if path is None:
            continue
        if not os.path.isdir(path):
            raise InvalidPathError(f"The {label} directory does not exist: {path}")
        if is_inside(path, args.output):
            raise InvalidPathError(
                f"The output directory ({args.output}) must not be the same as, or contain, "
                f"the {label} directory ({path})"
            )
    if os.path.exists(args.output) and not os.path.isdir(args.output):
        raise InvalidPathError(f"The output path exists and is not a directory: {args.output}")


def _resolve_settings(args: argparse.Namespace) -> Settings:
    """Merge defaults, config file and CLI flags (``--threshold`` maps per command)."""
    if args.config and not os.path.isfile(args.config):
        raise ConfigError(f"Config file not found: {args.config}")
    return load_settings(
        args.config or DEFAULT_CONFIG,  # the implicit ./config.ini may be absent
        confidence=args.confidence,
        cluster_threshold=args.threshold if args.command == "sort" else None,
        match_threshold=args.threshold if args.command == "find" else None,
    )


def _run(args: argparse.Namespace, settings: Settings) -> None:
    """Dispatch to the selected command (imports deferred for fast --help)."""
    if args.command == "sort":
        from .pipeline import process_images

        logger.info("Starting face sorting pipeline...")
        process_images(
            source_dir=args.source,
            output_dir=args.output,
            cluster_threshold=settings.cluster_threshold,
            confidence=settings.confidence,
        )
    else:
        from .finder import find_matches

        logger.info("Starting face finder...")
        find_matches(
            source_dir=args.source,
            output_dir=args.output,
            reference_dir=args.reference,
            threshold=settings.match_threshold,
            confidence=settings.confidence,
        )


def main(argv: list[str] | None = None) -> int:
    """
    Run the CLI.

    Returns:
        0 on success, 1 if the run failed (e.g. no images or no faces found), 2 for
        invalid arguments, paths or configuration.
    """
    args = build_parser().parse_args(argv)
    try:
        _check_paths(args)
        settings = _resolve_settings(args)
    except (ConfigError, InvalidPathError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    handlers: list[logging.Handler] = []
    try:
        try:
            os.makedirs(args.output, exist_ok=True)
            handlers = setup_logging(os.path.join(args.output, LOG_NAME), args.verbose)
        except OSError as exc:
            print(f"error: cannot use output directory {args.output}: {exc}", file=sys.stderr)
            return 2
        _run(args, settings)
        logger.info("Processing complete. Runtime log: %s", os.path.join(args.output, LOG_NAME))
    except FaceSorterError as exc:
        logger.error("%s", exc)  # goes to stderr and the runtime log
        return 2 if isinstance(exc, InvalidPathError) else 1
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except Exception as exc:  # last resort: a clean one-liner, traceback only with -v
        logger.error("Unexpected %s: %s", type(exc).__name__, exc, exc_info=args.verbose)
        return 1
    finally:
        teardown_logging(handlers)
    return 0
