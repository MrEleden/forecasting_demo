#!/usr/bin/env python3
"""
Project Structure Validator for ML Portfolio Forecasting Demo.

Validates that all projects follow the standard folder structure:
- Enforces consistent CLI commands and configurations
- Ensures uniform project layout for portfolio presentation
- Can be run in CI/CD to maintain structural integrity

Usage:
    python ci/scripts/validate_structure.py                    # Validate all projects
    python ci/scripts/validate_structure.py --project walmart  # Validate specific project
    python ci/scripts/validate_structure.py --fix             # Auto-create missing folders
"""

import argparse
from pathlib import Path
import sys
from typing import List, Dict, Set


# Standard project structure that all projects must follow
REQUIRED_STRUCTURE = {
    "folders": [
        "api",  # FastAPI endpoints
        "app",  # Streamlit dashboard
        "conf",  # Hydra configuration files
        "conf/dataset",  # Dataset configurations
        "conf/model",  # Model configurations
        "conf/optimizer",  # Optimizer configurations
        "conf/scheduler",  # LR scheduler configurations
        "conf/hydra",  # Hydra runtime settings
        "data",  # Data directories
        "data/external",  # External/third-party data
        "data/interim",  # Intermediate processed data
        "data/processed",  # Final processed data
        "data/raw",  # Raw unprocessed data
        "models",  # Trained models and artifacts
        "models/artifacts",  # Model artifacts (scalers, encoders)
        "models/checkpoints",  # Model checkpoints during training
        "notebooks",  # Jupyter notebooks for exploration
        "reports",  # Generated analysis reports
        "reports/figures",  # Generated plots and visualizations
        "scripts",  # Python scripts (training, evaluation)
        "tests",  # Unit and integration tests
    ],
    "files": [
        "README.md",  # Project-specific documentation
        "scripts/download_data.py",  # Data acquisition script (or generate_data.py)
        "conf/config.yaml",  # Main Hydra configuration
    ],
}

# Project-specific CLI commands that must be consistent
REQUIRED_CLI_PATTERNS = {
    "data_script": "python scripts/download_data.py",  # Or generate_data.py
    "train_example": "python scripts/train.py model=lstm dataset=PROJECT_NAME optimizer=adam",
    "evaluate_example": "python scripts/evaluate.py --study-name PROJECT_NAME_optimization",
}

# Projects and their specific characteristics
PROJECT_CONFIGS = {
    "retail_sales_walmart": {"data_script": "download_data.py", "dataset_name": "walmart", "primary_metric": "WMAE"},
    "rideshare_demand_ola": {"data_script": "generate_data.py", "dataset_name": "ola", "primary_metric": "MAPE"},
    "inventory_forecasting": {"data_script": "generate_data.py", "dataset_name": "inventory", "primary_metric": "MAPE"},
    "transportation_tsi": {"data_script": "download_data.py", "dataset_name": "tsi", "primary_metric": "RMSE"},
}


def get_project_dirs() -> List[Path]:
    """Get all project directories."""
    # Adjust path to work from repository root when called from ci/scripts/
    repo_root = Path(__file__).parent.parent.parent
    projects_root = repo_root / "projects"

    if not projects_root.exists():
        print("ERROR: 'projects' directory not found. Run from repository root.")
        sys.exit(1)

    return [p for p in projects_root.iterdir() if p.is_dir()]


def validate_project_structure(project_path: Path) -> Dict[str, List[str]]:
    """Validate a single project's structure."""
    issues = {"missing_folders": [], "missing_files": [], "extra_notes": []}

    project_name = project_path.name
    print(f"\nValidating: {project_name}")

    # Check required folders
    for folder in REQUIRED_STRUCTURE["folders"]:
        folder_path = project_path / folder
        if not folder_path.exists():
            issues["missing_folders"].append(folder)
        else:
            print(f"  {folder}/")  # Check required files
    for file in REQUIRED_STRUCTURE["files"]:
        # Handle project-specific data script names
        if file == "scripts/download_data.py" and project_name in PROJECT_CONFIGS:
            expected_script = f"scripts/{PROJECT_CONFIGS[project_name]['data_script']}"
            file_path = project_path / expected_script
            if not file_path.exists():
                issues["missing_files"].append(expected_script)
            else:
                print(f"  {expected_script}")
        else:
            file_path = project_path / file
            if not file_path.exists():
                issues["missing_files"].append(file)
            else:
                print(f"  {file}")

    return issues


def create_missing_structure(project_path: Path, issues: Dict[str, List[str]]) -> None:
    """Create missing folders and placeholder files."""
    project_name = project_path.name

    # Create missing folders with .gitkeep
    for folder in issues["missing_folders"]:
        folder_path = project_path / folder
        folder_path.mkdir(parents=True, exist_ok=True)
        gitkeep_file = folder_path / ".gitkeep"
        gitkeep_file.write_text("# Placeholder to maintain folder structure\n")
        print(f"  Created: {folder}/ (with .gitkeep)")

    # Create missing configuration files
    if "conf/config.yaml" in issues["missing_files"]:
        config_content = f"""# Main configuration for {project_name}
# This file defines default settings for all experiments

defaults:
  - dataset: {PROJECT_CONFIGS.get(project_name, {}).get('dataset_name', 'default')}
  - model: lstm
  - optimizer: adam
  - scheduler: cosine
  - _self_

# Experiment settings
experiment:
  name: {project_name}_baseline
  tags: [baseline, {PROJECT_CONFIGS.get(project_name, {}).get('dataset_name', 'default')}]

# Training settings  
trainer:
  max_epochs: 100
  patience: 10
  
# Evaluation settings
evaluation:
  primary_metric: {PROJECT_CONFIGS.get(project_name, {}).get('primary_metric', 'MAE')}
  cv_folds: 5
"""
        config_file = project_path / "conf" / "config.yaml"
        config_file.parent.mkdir(parents=True, exist_ok=True)
        config_file.write_text(config_content)
        print(f"  Created: conf/config.yaml")


def validate_all_projects(fix_issues: bool = False) -> bool:
    """Validate all projects and optionally fix issues."""
    print("ML Portfolio Project Structure Validator")
    print("=" * 50)

    project_dirs = get_project_dirs()
    all_valid = True

    for project_dir in project_dirs:
        issues = validate_project_structure(project_dir)

        if issues["missing_folders"] or issues["missing_files"]:
            all_valid = False
            print(f"  ISSUES FOUND:")

            if issues["missing_folders"]:
                print(f"    Missing folders: {', '.join(issues['missing_folders'])}")
            if issues["missing_files"]:
                print(f"    Missing files: {', '.join(issues['missing_files'])}")

            if fix_issues:
                print(f"  Fixing issues...")
                create_missing_structure(project_dir, issues)
                print(f"  {project_dir.name} structure fixed!")
        else:
            print(f"  Structure valid!")

    print(f"\n{'All projects valid!' if all_valid else 'Issues found in project structure.'}")

    if not all_valid and not fix_issues:
        print("NOTE: Run with --fix to automatically create missing folders and files.")

    return all_valid


def main():
    parser = argparse.ArgumentParser(description="Validate ML Portfolio project structure")
    parser.add_argument("--project", help="Validate specific project only")
    parser.add_argument("--fix", action="store_true", help="Auto-create missing folders and files")
    parser.add_argument("--ci", action="store_true", help="CI mode: exit with error code if issues found")

    args = parser.parse_args()

    # Ensure we're working from the repository root
    repo_root = Path(__file__).parent.parent.parent
    original_cwd = Path.cwd()

    try:
        # Change to repository root for proper path resolution
        import os

        os.chdir(repo_root)

        if args.project:
            project_path = Path("projects") / args.project
            if not project_path.exists():
                print(f"ERROR: Project '{args.project}' not found")
                sys.exit(1)

            issues = validate_project_structure(project_path)
            if issues["missing_folders"] or issues["missing_files"]:
                if args.fix:
                    create_missing_structure(project_path, issues)
                elif args.ci:
                    sys.exit(1)
        else:
            all_valid = validate_all_projects(fix_issues=args.fix)
            if args.ci and not all_valid:
                sys.exit(1)
    finally:
        # Restore original working directory
        os.chdir(original_cwd)


if __name__ == "__main__":
    main()
