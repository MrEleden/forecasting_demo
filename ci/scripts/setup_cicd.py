#!/usr/bin/env python3
"""
Simple CI/CD Setup Script for ML Portfolio

This script sets up basic CI/CD for the ML Portfolio project.

Usage:
    python ci/scripts/setup_cicd.py
"""

import shutil
import sys
from pathlib import Path


def setup_workflows():
    """Copy GitHub Actions workflows."""
    print("Setting up CI/CD workflows...")

    repo_root = Path.cwd()
    while not (repo_root / ".git").exists() and repo_root != repo_root.parent:
        repo_root = repo_root.parent

    workflows_source = repo_root / "ci" / "github-actions"
    workflows_dest = repo_root / ".github" / "workflows"

    workflows_dest.mkdir(parents=True, exist_ok=True)

    for workflow_file in workflows_source.glob("*.yml"):
        dest_file = workflows_dest / workflow_file.name
        shutil.copy2(workflow_file, dest_file)
        print(f"Copied: {workflow_file.name}")

    print("Setup complete!")
    print("Next steps:")
    print("1. Commit workflows: git add .github/workflows/")
    print("2. Push to trigger CI/CD: git push")
    print("3. Deploy locally: ./ci/scripts/deploy.sh deploy")


if __name__ == "__main__":
    setup_workflows()
