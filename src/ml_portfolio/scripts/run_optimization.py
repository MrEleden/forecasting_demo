# ==============================================================================
# Walmart Model Optimization Runner
# ==============================================================================
#
# Runs Optuna hyperparameter optimization for multiple models on Walmart dataset.
# Each model uses its own search space configuration from conf/optuna/
#
# Usage:
#   python run_optimization.py                    # Run all models with default trials (50)
#   python run_optimization.py --trials 20        # Run all models with 20 trials each
#   python run_optimization.py --models lightgbm xgboost  # Run specific models only
#   python run_optimization.py --models lightgbm --trials 100  # 100 trials for LightGBM only
#
# ==============================================================================

import argparse
import subprocess
import sys

# Available models with their optuna config names
AVAILABLE_MODELS = {
    # Statistical/Tree-based models
    "lightgbm": "lightgbm",
    "xgboost": "xgboost",
    "catboost": "catboost",
    "random_forest": "random_forest",
    # PyTorch deep learning models
    "lstm": "lstm",
    "tcn": "tcn",
    "transformer": "transformer",
    # Ensemble models
    "voting": "voting",
    "stacking": "stacking",
}


def run_optimization(model: str, optuna_config: str, n_trials: int, dry_run: bool = False):
    """
    Run Optuna optimization for a specific model.

    Args:
        model: Model name (e.g., 'lightgbm')
        optuna_config: Optuna config name (e.g., 'lightgbm')
        n_trials: Number of Optuna trials to run
        dry_run: If True, print command without executing
    """
    cmd = [
        "python",
        "src/ml_portfolio/training/train.py",
        "--multirun",
        "--config-name",
        "walmart",
        f"model={model}",
        "use_optuna=true",
        "hydra/sweeper=optuna",
        f"+optuna={optuna_config}",
        f"hydra.sweeper.n_trials={n_trials}",
    ]

    print(f"\n{'='*80}")
    print(f"MODEL: {model.upper()}")
    print(f"Trials: {n_trials}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}\n")

    if dry_run:
        print("DRY RUN - Command not executed\n")
        return 0

    try:
        result = subprocess.run(cmd, check=True)
        print(f"\n✓ {model} optimization completed successfully\n")
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {model} optimization failed with return code {e.returncode}\n")
        return e.returncode
    except KeyboardInterrupt:
        print(f"\n✗ {model} optimization interrupted by user\n")
        return 130


def main():
    parser = argparse.ArgumentParser(
        description="Run Optuna hyperparameter optimization for Walmart forecasting models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all models with default 50 trials each
  python run_optimization.py

  # Run all models with 20 trials each
  python run_optimization.py --trials 20

  # Run only LightGBM and XGBoost (tree-based models)
  python run_optimization.py --models lightgbm xgboost

  # Run PyTorch deep learning models
  python run_optimization.py --models lstm tcn transformer

  # Run LSTM with 100 trials
  python run_optimization.py --models lstm --trials 100

  # Dry run to see commands without executing
  python run_optimization.py --dry-run
        """,
    )

    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(AVAILABLE_MODELS.keys()),
        default=list(AVAILABLE_MODELS.keys()),
        help="Models to optimize (default: all models)",
    )

    parser.add_argument("--trials", type=int, default=50, help="Number of Optuna trials per model (default: 50)")

    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them")

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("WALMART MODEL OPTIMIZATION")
    print("=" * 80)
    print(f"Models: {', '.join(args.models)}")
    print(f"Trials per model: {args.trials}")
    print(f"Total trials: {len(args.models) * args.trials}")
    if args.dry_run:
        print("Mode: DRY RUN (commands will not be executed)")
    print("=" * 80)

    results = {}

    for model in args.models:
        optuna_config = AVAILABLE_MODELS[model]
        return_code = run_optimization(model, optuna_config, args.trials, args.dry_run)
        results[model] = return_code

        # If optimization failed and not in dry run, ask user if they want to continue
        if return_code != 0 and not args.dry_run:
            response = input(f"\n{model} optimization failed. Continue with remaining models? [y/N]: ")
            if response.lower() != "y":
                print("\nStopping optimization run.")
                break

    # Print summary
    print("\n" + "=" * 80)
    print("OPTIMIZATION SUMMARY")
    print("=" * 80)

    for model, return_code in results.items():
        status = "✓ SUCCESS" if return_code == 0 else f"✗ FAILED (code {return_code})"
        print(f"{model:20s} {status}")

    print("=" * 80)

    # Return non-zero if any optimization failed
    if any(code != 0 for code in results.values()):
        print("\n⚠ Some optimizations failed. Check logs above for details.")
        return 1
    else:
        print("\n✓ All optimizations completed successfully!")
        print("\nNext steps:")
        print("  1. Check MLflow UI for results: mlflow ui")
        print("  2. Compare models in walmart_sales_forecasting experiment")
        print("  3. Select best model based on val_MAPE metric")
        return 0


if __name__ == "__main__":
    sys.exit(main())
