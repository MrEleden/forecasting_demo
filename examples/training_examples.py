"""
Example usage of the unified training script.

This script demonstrates how to use both normal training and Optuna optimization
from the single train.py script.
"""

import subprocess
from pathlib import Path


def run_command(command, description):
    """Run a command and print the result."""
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"COMMAND: {command}")
    print(f"{'='*60}")

    result = subprocess.run(command, shell=True, capture_output=True, text=True)

    if result.returncode == 0:
        print("✅ SUCCESS")
        if result.stdout:
            print("STDOUT:")
            print(result.stdout)
    else:
        print("❌ FAILED")
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        if result.stdout:
            print("STDOUT:")
            print(result.stdout)

    return result.returncode == 0


def main():
    """Run training examples."""

    # Ensure we're in the right directory
    project_root = Path(__file__).parent.parent
    print(f"Project root: {project_root}")

    # Examples for normal training (no optuna config)
    training_examples = [
        {
            "command": "python src/ml_portfolio/training/train.py model=arima dataset_factory=walmart",
            "description": "Normal ARIMA training - evaluates test set",
        },
        {
            "command": "python src/ml_portfolio/training/train.py model=random_forest "
            "dataset_factory=walmart training.max_epochs=1",
            "description": "Normal Random Forest training - evaluates test set",
        },
    ]

    # Examples for Optuna optimization (with Hydra sweeper)
    optuna_examples = [
        {
            "command": "python src/ml_portfolio/training/train.py -m hydra/sweeper=optuna "
            "hydra.sweeper.n_trials=5 model=arima dataset_factory=walmart "
            "'model.order=choice([1,1,1],[2,1,1],[1,1,2])'",
            "description": "ARIMA optimization - 5 trials using Hydra Optuna sweeper",
        },
        {
            "command": "python src/ml_portfolio/training/train.py -m hydra/sweeper=optuna "
            "hydra.sweeper.n_trials=3 model=random_forest dataset_factory=walmart "
            "'model.n_estimators=int(interval(50,200))' 'model.max_depth=int(interval(3,15))'",
            "description": "Random Forest optimization - 3 trials using Hydra Optuna sweeper",
        },
        {
            "command": "python src/ml_portfolio/training/train.py -m hydra/sweeper=optuna "
            "hydra.sweeper.n_trials=2 model=lstm dataset_factory=walmart "
            "'model.hidden_size=int(interval(32,128))' 'optimizer.lr=float(interval(0.001,0.01))'",
            "description": "LSTM optimization - 2 trials using Hydra Optuna sweeper",
        },
    ]

    print("🚀 UNIFIED TRAINING SCRIPT WITH HYDRA SWEEPER")
    print("=" * 60)
    print("Same train.py script handles both single runs and Optuna optimization!")
    print("Use 'hydra/sweeper=optuna' with -m flag for hyperparameter optimization.")

    # Run normal training examples
    print("\n📊 SINGLE RUN EXAMPLES:")
    print("These train models directly and evaluate on the test set.")

    for example in training_examples:
        success = run_command(example["command"], example["description"])
        if not success:
            print(f"⚠️  Example failed: {example['description']}")

    print("\n🔍 HYDRA OPTUNA SWEEPER EXAMPLES:")
    print("These use Hydra's built-in Optuna sweeper for hyperparameter optimization.")

    for example in optuna_examples:
        success = run_command(example["command"], example["description"])
        if not success:
            print(f"⚠️  Example failed: {example['description']}")

    print("\n✨ EXAMPLES COMPLETED!")
    print("\nHydra Sweeper Architecture:")
    print("• Single run: python train.py → Test evaluation")
    print("• Multirun: python train.py -m hydra/sweeper=optuna → Validation optimization → Best config")
    print("• Use 'choice([a,b,c])' for categorical parameters")
    print("• Use 'int(interval(min,max))' for integer ranges")
    print("• Use 'float(interval(min,max))' for float ranges")
    print("\nMLflow tracking:")
    print("• Single run: Full training run logged")
    print("• Multirun: Only individual trials logged (Hydra manages the sweep)")


if __name__ == "__main__":
    main()
