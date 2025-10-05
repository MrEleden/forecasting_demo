# Cleanup Scripts

This directory contains utility scripts for cleaning up training artifacts and runs.

## clean_runs.py

Removes all stored training runs and artifacts to free up disk space.

### What Gets Cleaned

- **MLflow runs** (`mlruns/`) - Experiment tracking data, metrics, parameters
- **Hydra outputs** (`outputs/`) - Configuration outputs and logs
- **Checkpoints** (`checkpoints/`) - Saved model checkpoints

### Usage

```bash
# Preview what will be deleted (dry run)
python clean_runs.py --dry-run

# Clean everything
python clean_runs.py

# Keep directory structure but clean contents
python clean_runs.py --keep-dirs

# Clean specific components only
python clean_runs.py --mlflow-only
python clean_runs.py --hydra-only
python clean_runs.py --checkpoints-only
```

### Examples

**1. Preview before cleaning:**
```bash
python clean_runs.py --dry-run
```
Output shows files, directories, and disk space that would be freed.

**2. Clean everything:**
```bash
python clean_runs.py
```
Removes all MLflow, Hydra, and checkpoint data.

**3. Clean but preserve directory structure:**
```bash
python clean_runs.py --keep-dirs
```
Useful if you want to keep the directories for future runs.

**4. Clean only MLflow runs:**
```bash
python clean_runs.py --mlflow-only
```
Removes only `mlruns/` directory, keeps outputs and checkpoints.

### When to Use

- **Before committing**: Clean up experiments before pushing code
- **Disk space**: Free up space when running low
- **Fresh start**: Reset environment for new experiment batches
- **Selective cleanup**: Clean only specific artifact types

### Safety Features

- **Dry run mode**: Preview deletions before executing
- **Statistics**: Shows file counts and disk space freed
- **Selective cleaning**: Clean only what you need
- **Error handling**: Graceful handling of permission errors

### Git Integration

All cleaned directories are in `.gitignore`:
```gitignore
logs/
outputs/
mlruns/
checkpoints/
```

This prevents accidentally committing large artifact files to version control.

### Example Output

```bash
$ python clean_runs.py --dry-run
================================================================================
DRY RUN - No files will be deleted
================================================================================

  MLflow Runs:
    - Would delete: 13916 files, 1892 directories
    - Would free: 1.14 GB

  Hydra Outputs:
    - Would delete: 1362 files, 699 directories
    - Would free: 4.66 MB

  Checkpoints:
    - Would delete: 2 files, 2 directories
    - Would free: 14.79 MB

================================================================================
SUMMARY (DRY RUN)
================================================================================
Total files: 15280
Total directories: 2593
Total space freed: 1.16 GB

Run without --dry-run to actually delete these files
```

### Notes

- Always use `--dry-run` first to verify what will be deleted
- Cleaned data cannot be recovered - ensure you've saved important models
- The script preserves your code and configuration files
- Project-specific data in `projects/*/data/` is not affected

---

**Last Updated**: October 3, 2025
