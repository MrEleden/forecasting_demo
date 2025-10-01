# Hydra Configuration for ML Portfolio

This directory contains Hydra configuration files for training various forecasting models.

## Structure

```
conf/
├── config.yaml              # Main configuration file
├── dataloader/              # DataLoader configurations
│   ├── simple.yaml          # SimpleDataLoader (sklearn models)
│   └── pytorch.yaml         # PyTorchDataLoader (deep learning)
├── dataset/                 # Dataset configurations
│   └── default.yaml         # Default time series dataset
├── model/                   # Model configurations
│   ├── random_forest.yaml   # Random Forest (sklearn)
│   ├── arima.yaml           # ARIMA (statsmodels)
│   └── lstm.yaml            # LSTM (PyTorch)
├── optimizer/               # Optimizer configurations (PyTorch)
│   ├── adam.yaml            # Adam optimizer
│   ├── adamw.yaml           # AdamW optimizer
│   └── sgd.yaml             # SGD optimizer
├── scheduler/               # LR scheduler configurations (PyTorch)
│   ├── step_lr.yaml         # Step decay
│   ├── cosine.yaml          # Cosine annealing
│   └── plateau.yaml         # Reduce on plateau
└── metrics/                 # Metrics configurations
    └── default.yaml         # Default regression metrics
```

## Usage

### Basic Training

```bash
# Train with default configuration (Random Forest + SimpleDataLoader)
python src/ml_portfolio/training/train.py

# Train with LSTM model
python src/ml_portfolio/training/train.py model=lstm dataloader=pytorch training.max_epochs=100

# Train with ARIMA
python src/ml_portfolio/training/train.py model=arima dataloader=simple training.max_epochs=1
```

### Configuration Override

```bash
# Override specific parameters
python src/ml_portfolio/training/train.py \
    model=lstm \
    model.hidden_size=256 \
    model.num_layers=3 \
    dataloader=pytorch \
    dataloader.batch_size=128 \
    training.max_epochs=50
```

### Multi-Run Experiments

```bash
# Sweep over multiple configurations
python src/ml_portfolio/training/train.py -m \
    model=lstm,random_forest \
    dataloader.batch_size=32,64,128 \
    training.max_epochs=50,100
```

## Configuration Patterns

### For Sklearn Models (Single-Shot Training)

```yaml
defaults:
  - dataloader: simple  # Returns all data in one batch
  - model: random_forest

training:
  max_epochs: 1  # Single training pass
  early_stopping: false
```

### For PyTorch Models (Iterative Training)

```yaml
defaults:
  - dataloader: pytorch  # Mini-batch iteration
  - model: lstm
  - optimizer: adam
  - scheduler: step_lr

training:
  max_epochs: 100  # Multiple epochs
  early_stopping: true
  patience: 10
```

## Creating Custom Configurations

### New Model Configuration

Create `conf/model/my_model.yaml`:

```yaml
_target_: my_module.MyModel

# Model parameters
param1: value1
param2: value2
```

### New DataLoader Configuration

Create `conf/dataloader/my_loader.yaml`:

```yaml
_target_: ml_portfolio.data.loaders.MyLoader

batch_size: 64
shuffle: true
```

## Hydra Features

- **Composition**: Combine configs with `defaults` directive
- **Override**: Command-line overrides with `key=value`
- **Sweep**: Multi-run experiments with `-m` flag
- **Instantiation**: Automatic object creation via `_target_`
- **Output**: Organized outputs with timestamp directories

## Best Practices

1. **Sklearn Models**: Use `SimpleDataLoader` + `max_epochs=1`
2. **PyTorch Models**: Use `PyTorchDataLoader` + `max_epochs>1` + optimizer/scheduler
3. **Naming Convention**: Use descriptive names for configs (e.g., `lstm_deep.yaml`)
4. **Documentation**: Add comments explaining parameter choices
5. **Version Control**: Commit config files to track experiment settings

## References

- [Hydra Documentation](https://hydra.cc/)
- [Config Composition](https://hydra.cc/docs/advanced/defaults_list/)
- [Object Instantiation](https://hydra.cc/docs/advanced/instantiate_objects/overview/)
