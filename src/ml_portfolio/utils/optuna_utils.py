"""
Optuna integration for hyperparameter optimization with Hydra and MLflow.

This module provides utilities to integrate Optuna with the training pipeline,
allowing for automated hyperparameter tuning while maintaining compatibility
with Hydra configuration management and MLflow experiment tracking.
"""

import logging
from typing import Any, Callable, Dict

import hydra
import mlflow
import optuna
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


class HydraOptunaIntegration:
    """
    Integration between Optuna, Hydra, and MLflow for hyperparameter optimization.

    This class manages:
    - Optuna study creation and configuration
    - Hyperparameter space definition from Hydra configs
    - Integration with MLflow for tracking optimization results
    - Proper train/val/test split handling
    """

    def __init__(self, cfg: DictConfig, objective_function: Callable):
        """
        Initialize the Optuna integration.

        Args:
            cfg: Hydra configuration containing optuna and hyperparam_space configs
            objective_function: Function to optimize (should take trial and return metric)
        """
        self.cfg = cfg
        self.objective_function = objective_function
        self.study = None
        self.best_params = None

    def create_study(self) -> optuna.Study:
        """Create and configure Optuna study."""
        # Get study configuration
        study_cfg = self.cfg.optuna

        # Create sampler
        sampler = hydra.utils.instantiate(study_cfg.sampler)

        # Create pruner if specified
        pruner = None
        if hasattr(study_cfg, "pruner") and study_cfg.pruner:
            pruner = hydra.utils.instantiate(study_cfg.pruner)

        # Create study
        self.study = optuna.create_study(
            study_name=study_cfg.study_name,
            storage=study_cfg.storage,
            direction=study_cfg.direction,
            sampler=sampler,
            pruner=pruner,
            load_if_exists=True,
        )

        logger.info(f"Created Optuna study: {study_cfg.study_name}")
        logger.info(f"Direction: {study_cfg.direction}")
        logger.info(f"Storage: {study_cfg.storage}")

        return self.study

    def suggest_hyperparameters(self, trial: optuna.Trial, model_name: str) -> Dict[str, Any]:
        """
        Suggest hyperparameters based on the search space configuration.

        Args:
            trial: Optuna trial object
            model_name: Name of the model (e.g., 'arima', 'lstm', 'random_forest')

        Returns:
            Dictionary of suggested hyperparameters
        """
        # Get hyperparameter space config
        hyperparam_cfg = getattr(self.cfg, "hyperparam_space", {})

        if model_name == "arima":
            return self._suggest_arima_params(trial, hyperparam_cfg)
        elif model_name == "random_forest":
            return self._suggest_rf_params(trial, hyperparam_cfg)
        elif model_name == "lstm":
            return self._suggest_lstm_params(trial, hyperparam_cfg)
        else:
            logger.warning(f"No hyperparameter space defined for model: {model_name}")
            return {}

    def _suggest_arima_params(self, trial: optuna.Trial, cfg: DictConfig) -> Dict[str, Any]:
        """Suggest ARIMA hyperparameters."""
        params = {}

        # ARIMA order (p, d, q)
        p = trial.suggest_int("p", cfg.get("p_min", 0), cfg.get("p_max", 5))
        d = trial.suggest_int("d", cfg.get("d_min", 0), cfg.get("d_max", 2))
        q = trial.suggest_int("q", cfg.get("q_min", 0), cfg.get("q_max", 5))
        params["order"] = [p, d, q]

        # Seasonal order (P, D, Q, s)
        P = trial.suggest_int("seasonal_p", cfg.get("seasonal_p_min", 0), cfg.get("seasonal_p_max", 2))
        D = trial.suggest_int("seasonal_d", cfg.get("seasonal_d_min", 0), cfg.get("seasonal_d_max", 1))
        Q = trial.suggest_int("seasonal_q", cfg.get("seasonal_q_min", 0), cfg.get("seasonal_q_max", 2))
        s = cfg.get("seasonal_s", 12)
        params["seasonal_order"] = [P, D, Q, s]

        # Other parameters
        if "trend_options" in cfg:
            params["trend"] = trial.suggest_categorical("trend", cfg.trend_options)

        if "enforce_stationarity" in cfg:
            params["enforce_stationarity"] = trial.suggest_categorical("enforce_stationarity", cfg.enforce_stationarity)

        if "enforce_invertibility" in cfg:
            params["enforce_invertibility"] = trial.suggest_categorical(
                "enforce_invertibility", cfg.enforce_invertibility
            )

        return params

    def _suggest_rf_params(self, trial: optuna.Trial, cfg: DictConfig) -> Dict[str, Any]:
        """Suggest Random Forest hyperparameters."""
        params = {}

        # Number of estimators
        params["n_estimators"] = trial.suggest_int(
            "n_estimators",
            cfg.get("n_estimators_min", 50),
            cfg.get("n_estimators_max", 500),
            step=cfg.get("n_estimators_step", 50),
        )

        # Tree depth
        params["max_depth"] = trial.suggest_int("max_depth", cfg.get("max_depth_min", 3), cfg.get("max_depth_max", 20))

        # Feature sampling
        if "max_features_options" in cfg:
            params["max_features"] = trial.suggest_categorical("max_features", cfg.max_features_options)

        # Sample sampling
        params["min_samples_split"] = trial.suggest_int(
            "min_samples_split", cfg.get("min_samples_split_min", 2), cfg.get("min_samples_split_max", 20)
        )

        params["min_samples_leaf"] = trial.suggest_int(
            "min_samples_leaf", cfg.get("min_samples_leaf_min", 1), cfg.get("min_samples_leaf_max", 10)
        )

        # Other parameters
        if "bootstrap" in cfg:
            params["bootstrap"] = trial.suggest_categorical("bootstrap", cfg.bootstrap)

        if "random_state" in cfg:
            params["random_state"] = cfg.random_state

        return params

    def _suggest_lstm_params(self, trial: optuna.Trial, cfg: DictConfig) -> Dict[str, Any]:
        """Suggest LSTM hyperparameters."""
        params = {}

        # Network architecture
        if "hidden_size_options" in cfg:
            params["hidden_size"] = trial.suggest_categorical("hidden_size", cfg.hidden_size_options)

        params["num_layers"] = trial.suggest_int(
            "num_layers", cfg.get("num_layers_min", 1), cfg.get("num_layers_max", 4)
        )

        params["dropout"] = trial.suggest_float(
            "dropout", cfg.get("dropout_min", 0.0), cfg.get("dropout_max", 0.5), step=cfg.get("dropout_step", 0.1)
        )

        # Training parameters
        if "batch_size_options" in cfg:
            params["batch_size"] = trial.suggest_categorical("batch_size", cfg.batch_size_options)

        # Learning rate (log scale if specified)
        if cfg.get("learning_rate_log", False):
            params["learning_rate"] = trial.suggest_float(
                "learning_rate", cfg.get("learning_rate_min", 1e-5), cfg.get("learning_rate_max", 1e-2), log=True
            )
        else:
            params["learning_rate"] = trial.suggest_float(
                "learning_rate", cfg.get("learning_rate_min", 1e-5), cfg.get("learning_rate_max", 1e-2)
            )

        # Other parameters
        if "bidirectional" in cfg:
            params["bidirectional"] = trial.suggest_categorical("bidirectional", cfg.bidirectional)

        return params

    def optimize(self) -> Dict[str, Any]:
        """
        Run hyperparameter optimization.

        Returns:
            Best parameters found during optimization
        """
        if self.study is None:
            self.create_study()

        # Setup MLflow for optimization tracking
        if self.cfg.optuna.get("log_to_mlflow", False):
            experiment_name = (
                f"{self.cfg.mlflow.experiment_name}{self.cfg.optuna.get('mlflow_experiment_suffix', '_optuna')}"
            )
            mlflow.set_experiment(experiment_name)

        # Run optimization
        logger.info(f"Starting optimization with {self.cfg.optuna.n_trials} trials")

        self.study.optimize(
            self.objective_function,
            n_trials=self.cfg.optuna.n_trials,
            timeout=self.cfg.optuna.get("timeout"),
            n_jobs=self.cfg.optuna.get("n_jobs", 1),
        )

        # Get best parameters
        self.best_params = self.study.best_params

        logger.info("Optimization completed!")
        logger.info(f"Best value: {self.study.best_value}")
        logger.info(f"Best parameters: {self.best_params}")

        return self.best_params

    def get_best_trial_results(self) -> Dict[str, Any]:
        """Get comprehensive results from the best trial."""
        if self.study is None or self.best_params is None:
            raise ValueError("Optimization must be run before getting best results")

        best_trial = self.study.best_trial

        return {
            "best_value": self.study.best_value,
            "best_params": self.best_params,
            "trial_number": best_trial.number,
            "trial_state": best_trial.state,
            "duration": best_trial.duration,
            "user_attrs": best_trial.user_attrs,
            "system_attrs": best_trial.system_attrs,
        }


def create_optuna_objective(cfg: DictConfig, train_func: Callable) -> Callable:
    """
    Create an objective function for Optuna optimization.

    Args:
        cfg: Hydra configuration
        train_func: Training function that takes config and returns validation metric

    Returns:
        Objective function for Optuna
    """

    def objective(trial: optuna.Trial) -> float:
        """
        Objective function for Optuna optimization.

        This function:
        1. Suggests hyperparameters based on the search space
        2. Updates the configuration with suggested parameters
        3. Trains the model with these parameters
        4. Returns the validation metric for optimization
        """
        # Create integration instance
        integration = HydraOptunaIntegration(cfg, None)

        # Extract model name from target
        model_name = cfg.model._target_.split(".")[-1].lower()
        model_name = model_name.replace("wrapper", "").replace("forecaster", "")

        # Suggest hyperparameters
        suggested_params = integration.suggest_hyperparameters(trial, model_name)

        # Create a copy of the config and update with suggested parameters
        trial_cfg = OmegaConf.create(OmegaConf.to_yaml(cfg))

        # Update model parameters
        for key, value in suggested_params.items():
            OmegaConf.set(trial_cfg.model, key, value)

        # Log trial info to MLflow if enabled
        if cfg.optuna.get("log_to_mlflow", False):
            with mlflow.start_run(nested=True):
                mlflow.log_params(suggested_params)
                mlflow.set_tag("optuna_trial", trial.number)

                # Train and get validation metric
                val_metric = train_func(trial_cfg, trial)

                mlflow.log_metric(cfg.optuna.optimize_metric, val_metric)

                return val_metric
        else:
            return train_func(trial_cfg, trial)

    return objective
