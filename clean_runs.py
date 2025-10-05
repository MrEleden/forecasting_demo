"""
Clean all stored training runs and artifacts.

This script removes:
- MLflow runs (mlruns/)
- Hydra outputs (outputs/)
- Model checkpoints (checkpoints/)

Usage:
    python clean_runs.py [--dry-run] [--keep-dirs]

Options:
    --dry-run      Show what would be deleted without actually deleting
    --keep-dirs    Keep the empty directories after cleanup
    --mlflow-only  Only clean MLflow runs
    --hydra-only   Only clean Hydra outputs
    --checkpoints-only  Only clean checkpoints

Examples:
    python clean_runs.py                    # Clean everything
    python clean_runs.py --dry-run          # Preview what will be deleted
    python clean_runs.py --keep-dirs        # Clean but keep directory structure
    python clean_runs.py --mlflow-only      # Only clean MLflow
"""

import argparse
import shutil
from pathlib import Path


def get_dir_size(path: Path) -> int:
    """Calculate total size of directory in bytes."""
    total = 0
    try:
        for item in path.rglob("*"):
            if item.is_file():
                total += item.stat().st_size
    except (PermissionError, OSError):
        pass
    return total


def format_size(bytes_size: int) -> str:
    """Format bytes to human-readable size."""
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} TB"


def count_items(path: Path) -> tuple[int, int]:
    """Count files and directories in path."""
    files = 0
    dirs = 0
    try:
        for item in path.rglob("*"):
            if item.is_file():
                files += 1
            elif item.is_dir():
                dirs += 1
    except (PermissionError, OSError):
        pass
    return files, dirs


def clean_directory(path: Path, name: str, dry_run: bool = False, keep_dirs: bool = False) -> tuple[int, int, int]:
    """
    Clean a directory and return statistics.

    Args:
        path: Directory to clean
        name: Display name for logging
        dry_run: If True, don't actually delete anything
        keep_dirs: If True, keep the empty directory structure

    Returns:
        Tuple of (files_deleted, dirs_deleted, bytes_freed)
    """
    if not path.exists():
        print(f"  {name}: Not found (skipping)")
        return 0, 0, 0

    # Get statistics before deletion
    size = get_dir_size(path)
    files, dirs = count_items(path)

    if dry_run:
        print(f"  {name}:")
        print(f"    - Would delete: {files} files, {dirs} directories")
        print(f"    - Would free: {format_size(size)}")
        return files, dirs, size

    # Actual deletion
    try:
        if keep_dirs:
            # Remove only files and subdirectories, keep the root directory
            for item in path.iterdir():
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
            print(f"  {name}:")
            print(f"    - Deleted: {files} files, {dirs} directories")
            print(f"    - Freed: {format_size(size)}")
            print(f"    - Kept directory: {path}")
        else:
            # Remove everything including the root directory
            shutil.rmtree(path)
            print(f"  {name}:")
            print(f"    - Deleted: {files} files, {dirs} directories")
            print(f"    - Freed: {format_size(size)}")
            print(f"    - Removed directory: {path}")

        return files, dirs, size

    except Exception as e:
        print(f"  {name}: Error during cleanup - {e}")
        return 0, 0, 0


def main():
    """Main cleanup function."""
    parser = argparse.ArgumentParser(
        description="Clean all stored training runs and artifacts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without actually deleting",
    )
    parser.add_argument(
        "--keep-dirs",
        action="store_true",
        help="Keep the empty directories after cleanup",
    )
    parser.add_argument("--mlflow-only", action="store_true", help="Only clean MLflow runs")
    parser.add_argument("--hydra-only", action="store_true", help="Only clean Hydra outputs")
    parser.add_argument("--checkpoints-only", action="store_true", help="Only clean checkpoints")

    args = parser.parse_args()

    # Get repository root
    repo_root = Path(__file__).parent

    # Define directories to clean
    mlflow_dir = repo_root / "mlruns"
    hydra_dir = repo_root / "outputs"
    checkpoints_dir = repo_root / "checkpoints"

    # Determine what to clean based on flags
    clean_mlflow = not (args.hydra_only or args.checkpoints_only)
    clean_hydra = not (args.mlflow_only or args.checkpoints_only)
    clean_checkpoints = not (args.mlflow_only or args.hydra_only)

    print("=" * 80)
    if args.dry_run:
        print("DRY RUN - No files will be deleted")
    else:
        print("CLEANING STORED RUNS")
    print("=" * 80)
    print()

    # Track totals
    total_files = 0
    total_dirs = 0
    total_bytes = 0

    # Clean MLflow runs
    if clean_mlflow:
        files, dirs, size = clean_directory(mlflow_dir, "MLflow Runs", args.dry_run, args.keep_dirs)
        total_files += files
        total_dirs += dirs
        total_bytes += size
        print()

    # Clean Hydra outputs
    if clean_hydra:
        files, dirs, size = clean_directory(hydra_dir, "Hydra Outputs", args.dry_run, args.keep_dirs)
        total_files += files
        total_dirs += dirs
        total_bytes += size
        print()

    # Clean checkpoints
    if clean_checkpoints:
        files, dirs, size = clean_directory(checkpoints_dir, "Checkpoints", args.dry_run, args.keep_dirs)
        total_files += files
        total_dirs += dirs
        total_bytes += size
        print()

    # Print summary
    print("=" * 80)
    if args.dry_run:
        print("SUMMARY (DRY RUN)")
    else:
        print("SUMMARY")
    print("=" * 80)
    print(f"Total files: {total_files}")
    print(f"Total directories: {total_dirs}")
    print(f"Total space freed: {format_size(total_bytes)}")
    print()

    if args.dry_run:
        print("Run without --dry-run to actually delete these files")
    elif total_files > 0 or total_dirs > 0:
        print("Cleanup complete!")
    else:
        print("Nothing to clean")


if __name__ == "__main__":
    main()
