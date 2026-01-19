# Veritas Model Training

This directory contains the training infrastructure for the Veritas Sunrise ML model.

## 🚀 Quick Start

1. **Install dependencies:**
   ```bash
   ./setup.sh
   ```

2. **Start training:**
   ```bash
   ./run_training.sh start --quick
   ```

3. **Monitor progress:**
   ```bash
   ./run_training.sh monitor
   ```

## 📁 Files

### Training Scripts
- `train_sunrise.py` - Main training script with full ML pipeline
- `train_sunrise_efficient.py` - Memory-efficient training variant
- `train_sunrise_robust.py` - Robust training with error handling
- `sunrise_trainer.py` - Core trainer class with hyperparameter optimization
- `trainer.py` - Base trainer utilities

### Data Loading
- `massive_dataset_loader.py` - Loads multiple HuggingFace datasets
- `dataset_loader.py` - Base dataset loading utilities
- `curated_datasets.py` - Curated dataset configurations
- `discover_datasets.py` - Dataset discovery tools
- `discovered_datasets.json` - Registry of available datasets

### Feature Engineering
- `feature_extractor.py` - Extract linguistic features from text

### Automation
- `run_training.sh` - **Main automation script** (see below)
- `setup.sh` - Environment setup script
- `TRAINING_AUTOMATION.md` - Complete automation documentation

### Configuration
- `requirements.txt` - Python dependencies

## 🎯 Using the Automation

The `run_training.sh` script provides a complete interface for managing training:

```bash
# Get help
./run_training.sh help

# Start training (quick mode for testing)
./run_training.sh start --quick

# Start training (full mode)
./run_training.sh start

# Monitor training in real-time
./run_training.sh monitor

# Check current status
./run_training.sh status

# Check if training completed
./run_training.sh check

# Stop training if needed
./run_training.sh stop

# View full logs
./run_training.sh logs
```

## 📊 Training Modes

### Quick Mode (5-15 minutes)
```bash
./run_training.sh start --quick
```
- 5,000 samples per dataset
- 20 optimization trials
- Good for testing

### Standard Mode (30-60 minutes)
```bash
./run_training.sh start
```
- 50,000 samples per dataset
- 100 optimization trials
- Production-ready models

### Custom Configuration
```bash
./run_training.sh start --trials 200 --max-samples-per-dataset 100000
```

## 📖 Documentation

For detailed documentation, see:
- **[TRAINING_AUTOMATION.md](./TRAINING_AUTOMATION.md)** - Complete guide to training automation
- **Main README** - [../README.md](../README.md) - Project overview

## 🔧 Training Process

The training pipeline consists of:

1. **Dataset Loading** - Loads and validates multiple HuggingFace datasets
2. **Balancing** - Ensures equal human/AI sample distribution
3. **Feature Extraction** - Extracts 37 linguistic features
4. **Hyperparameter Optimization** - Uses Optuna for optimal parameters
5. **Training** - Trains Random Forest classifier
6. **Evaluation** - Validates with cross-validation and test set
7. **Export** - Saves model, scaler, and configurations

## 📦 Output

After training, models are saved to `./models/[ModelName]_[timestamp]/`:

- `model.pkl` - Trained ML model
- `scaler.pkl` - Feature scaler
- `training_receipt.json` - Complete training proof
- `metadata.json` - Model metadata
- `veritas_ml_config.js` - JavaScript config for web
- `veritas_ml_config.json` - JSON config
- `dataset_loading_report.json` - Dataset details

## 🌐 Cloud Training

For training on cloud instances:

```bash
# Clone and setup
git clone https://github.com/NagusameCS/Veritas.git
cd Veritas/training
./setup.sh

# Start training in background
./run_training.sh start

# Detach and monitor remotely
# (use screen or tmux for persistent sessions)
```

## ⚠️ Requirements

- Python 3.8+
- 4GB+ RAM (8GB+ recommended)
- Internet connection (for dataset downloads)
- ~2GB disk space for datasets

## 🐛 Troubleshooting

```bash
# Check training status
./run_training.sh status

# View error logs
./run_training.sh logs

# Clean restart
./run_training.sh stop
rm -f .training.pid sunrise_log.txt
./run_training.sh start --quick
```

## 📝 Example Workflow

```bash
# Terminal 1: Start training
cd training
./setup.sh
./run_training.sh start --quick

# Terminal 2: Monitor
cd training
./run_training.sh monitor

# Later: Check if complete
./run_training.sh check
```

## 💡 Tips

- Always test with `--quick` first
- Use `./run_training.sh monitor` in a separate terminal
- Training logs are saved to `sunrise_log.txt`
- Models include training receipts for reproducibility
- Check `./run_training.sh status` to monitor progress

## 🔗 Links

- **Main Project**: https://github.com/NagusameCS/Veritas
- **Live Demo**: https://nagusame.github.io/Veritas
- **npm Package**: https://www.npmjs.com/package/veritas-ai-detector
