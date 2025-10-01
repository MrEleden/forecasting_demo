# Scheduler Integration - Complete Guide

## Where Schedulers Are Defined

### 1. **Configuration Files** (`src/ml_portfolio/conf/scheduler/`)
Scheduler configurations are defined as YAML files with Hydra `_target_`:

```yaml
# scheduler/step_lr.yaml
_target_: torch.optim.lr_scheduler.StepLR
step_size: 30
gamma: 0.1
```

### 2. **train.py Script** (Instantiation)
Schedulers are instantiated in `train.py` (Section 5):

```python
# Section 5: Instantiate scheduler (for PyTorch models)
scheduler = None
if hasattr(cfg, "scheduler") and cfg.scheduler is not None and optimizer is not None:
    logger.info(f"Instantiating scheduler: {cfg.scheduler._target_}")
    scheduler = hydra.utils.instantiate(cfg.scheduler, optimizer=optimizer)
```

### 3. **TrainingEngine** (Usage)
Schedulers are used in the training loop inside `engine.py`:

```python
# Learning rate scheduler step (for PyTorch models)
if self.scheduler is not None:
    # Check if scheduler needs metric (ReduceLROnPlateau)
    if 'metrics' in self.scheduler.step.__code__.co_varnames:
        current_metric = val_metrics.get(self.monitor_metric, float("inf"))
        self.scheduler.step(current_metric)
    else:
        # Regular schedulers (StepLR, CosineAnnealingLR, etc.)
        self.scheduler.step()
```

## Integration Flow

```
config.yaml (scheduler: step_lr)
    ↓
train.py Section 5 (hydra.utils.instantiate)
    ↓
TrainingEngine.__init__ (stores scheduler)
    ↓
TrainingEngine._train() (calls scheduler.step())
    ↓
Learning rate updated each epoch
```

## Usage Examples

### Example 1: Sklearn Model (No Scheduler)
```bash
python src/ml_portfolio/training/train.py \
    model=random_forest \
    dataloader=simple \
    optimizer=null \
    scheduler=null \
    training.max_epochs=1
```

Config:
```yaml
defaults:
  - optimizer: null  # Not needed for sklearn
  - scheduler: null  # Not needed for sklearn
```

### Example 2: PyTorch Model with StepLR
```bash
python src/ml_portfolio/training/train.py \
    model=lstm \
    dataloader=pytorch \
    optimizer=adam \
    scheduler=step_lr \
    training.max_epochs=100
```

Config:
```yaml
defaults:
  - optimizer: adam
  - scheduler: step_lr

# Override scheduler parameters
scheduler:
  step_size: 20
  gamma: 0.5
```

### Example 3: PyTorch Model with ReduceLROnPlateau
```bash
python src/ml_portfolio/training/train.py \
    model=lstm \
    dataloader=pytorch \
    optimizer=adam \
    scheduler=plateau \
    training.max_epochs=100 \
    training.monitor_metric=val_loss
```

Config:
```yaml
defaults:
  - optimizer: adam
  - scheduler: plateau

training:
  monitor_metric: val_loss  # Plateau uses this metric
```

### Example 4: Optimizer/Scheduler Sweep
```bash
python src/ml_portfolio/training/train.py -m \
    model=lstm \
    optimizer=adam,adamw,sgd \
    scheduler=step_lr,cosine,plateau \
    training.max_epochs=50
```

## Scheduler Types and Behavior

### 1. **StepLR** (Regular Step)
- **Calls**: `scheduler.step()` (no arguments)
- **Behavior**: Decays LR every `step_size` epochs
- **Use case**: Simple periodic LR decay

### 2. **CosineAnnealingLR** (Regular Step)
- **Calls**: `scheduler.step()` (no arguments)
- **Behavior**: Cosine annealing from initial to min LR
- **Use case**: Smooth LR decay for transformers

### 3. **ReduceLROnPlateau** (Metric-based)
- **Calls**: `scheduler.step(metric)` (needs validation metric)
- **Behavior**: Reduces LR when metric plateaus
- **Use case**: Adaptive LR based on validation performance

## Key Code Locations

| Component | File | Lines | Purpose |
|-----------|------|-------|---------|
| Config definitions | `conf/scheduler/*.yaml` | N/A | Define scheduler parameters |
| Instantiation | `train.py` | ~125-135 | Create scheduler with Hydra |
| Storage | `engine.py` | ~45-55 | Store scheduler in engine |
| Usage | `engine.py` | ~175-190 | Call scheduler.step() each epoch |
| LR logging | `engine.py` | ~189-191 | Log current learning rate |

## Troubleshooting

### Issue: Scheduler not stepping
**Solution**: Ensure `optimizer` is not None (schedulers require optimizer)

### Issue: ReduceLROnPlateau error
**Solution**: Check that `training.monitor_metric` matches a metric in `val_metrics`

### Issue: Learning rate not changing
**Solution**: Verify scheduler.step() is called after each epoch (check verbose logs)

### Issue: Sklearn model with scheduler
**Solution**: Schedulers only work with PyTorch models - set `scheduler=null` for sklearn

## Best Practices

1. **Sklearn models**: Always set `optimizer=null scheduler=null`
2. **PyTorch models**: Always specify both `optimizer` and `scheduler`
3. **StepLR/Cosine**: Use for deterministic LR schedules
4. **ReduceLROnPlateau**: Use when unsure about optimal LR schedule
5. **Logging**: Enable `training.verbose=true` to see LR changes
6. **Sweeps**: Test multiple schedulers to find best for your model

## Architecture Benefits

✅ **Flexible**: Works with any PyTorch scheduler
✅ **Optional**: Automatically skips for sklearn models
✅ **Smart**: Detects ReduceLROnPlateau and passes metrics
✅ **Logged**: Learning rate changes logged each epoch
✅ **Config-driven**: All parameters in YAML files
✅ **Compatible**: Works with Hydra sweeps and Optuna
