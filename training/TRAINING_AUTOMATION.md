# Veritas Model Training Automation

This document describes how to use the automated training system for the Veritas Sunrise ML model.

## Overview

The training automation provides a simple interface to:
- **Start** model training in the background
- **Monitor** training progress in real-time
- **Check** training status and completion
- **Stop** training if necessary

## Quick Start

### 1. Start Training

Start training with default settings:
```bash
cd training
./run_training.sh start
```

Start training with quick mode (faster, for testing):
```bash
./run_training.sh start --quick
```

Start training with custom parameters:
```bash
./run_training.sh start --trials 50 --max-samples-per-dataset 10000
```

### 2. Monitor Training

In another terminal, monitor the training logs in real-time:
```bash
cd training
./run_training.sh monitor
```

Press `Ctrl+C` to exit monitoring (training continues in background).

### 3. Check Status

Check the current status of training:
```bash
./run_training.sh status
```

This shows:
- Whether training is running
- The process ID (PID)
- Recent log entries
- Latest accuracy (if available)

### 4. Check Completion

Check if training has completed successfully:
```bash
./run_training.sh check
```

This will show the final training results if training is complete.

### 5. Stop Training (if needed)

If you need to stop training:
```bash
./run_training.sh stop
```

## Training Script Arguments

The `train_sunrise.py` script accepts the following arguments:

| Argument | Description | Default |
|----------|-------------|---------|
| `--max-samples-per-dataset` | Maximum samples to load per dataset | 50,000 |
| `--min-priority` | Minimum dataset priority (1=highest, 3=lowest) | 2 |
| `--trials` | Number of Optuna optimization trials | 100 |
| `--quick` | Quick training mode (fewer samples and trials) | False |
| `--output-dir` | Output directory for models | ./models |
| `--name` | Model name | Sunrise |

### Example Usage

Quick training for testing:
```bash
./run_training.sh start --quick
```

Full training with custom trial count:
```bash
./run_training.sh start --trials 200
```

Custom output directory and name:
```bash
./run_training.sh start --output-dir ./my_models --name MySunriseModel
```

## Files Generated

During and after training, the following files are created:

### Temporary Files (During Training)
- `sunrise_log.txt` - Training logs
- `.training.pid` - Process ID of running training
- `.training_status` - Current status (running/completed/failed/stopped)

### Model Files (After Training)
Located in `./models/[ModelName]_[timestamp]/`:
- `model.pkl` - Trained ML model
- `scaler.pkl` - Feature scaler
- `training_receipt.json` - Complete training proof and metrics
- `metadata.json` - Model metadata
- `veritas_ml_config.js` - JavaScript configuration
- `veritas_ml_config.json` - JSON configuration for web
- `dataset_loading_report.json` - Dataset details

## Training Process

The training process consists of 4 main steps:

### Step 1: Loading Datasets
- Loads multiple HuggingFace AI detection datasets
- Applies dataset priority filtering
- Limits samples per dataset
- Reports loading statistics

### Step 2: Balancing Dataset
- Balances human and AI samples
- Ensures equal representation
- Prevents bias in training

### Step 3: Training Model
- Extracts features from text samples
- Performs hyperparameter optimization (if enabled)
- Trains ML model with best parameters
- Performs cross-validation
- Evaluates on test set

### Step 4: Saving Model
- Saves trained model and scaler
- Generates training receipts
- Creates verification hashes
- Exports configurations for web and CLI

## Monitoring Training Progress

### Real-time Monitoring
```bash
./run_training.sh monitor
```

### Periodic Status Checks
```bash
# In a loop
while true; do
    ./run_training.sh status
    sleep 60  # Check every minute
done
```

### View Full Logs
```bash
./run_training.sh logs
```

Or directly:
```bash
cat sunrise_log.txt
```

## Training Duration

Training time varies based on:
- Number of datasets loaded
- Samples per dataset
- Number of optimization trials
- Hardware capabilities

Typical durations:
- **Quick mode** (`--quick`): 5-15 minutes
- **Standard mode**: 30-60 minutes
- **Full optimization** (200+ trials): 1-3 hours

## Troubleshooting

### Training Won't Start
1. Check if training is already running:
   ```bash
   ./run_training.sh status
   ```

2. Check for stale PID files:
   ```bash
   rm .training.pid
   ```

3. Check dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Training Failed
1. Check the logs for errors:
   ```bash
   ./run_training.sh logs
   ```

2. Common issues:
   - Missing dependencies
   - Insufficient memory
   - Dataset download failures
   - Network connectivity issues

### Process Stuck
1. Check if process is actually running:
   ```bash
   ps aux | grep train_sunrise
   ```

2. Force stop if necessary:
   ```bash
   ./run_training.sh stop
   ```

## Cloud Deployment

To run training on a cloud instance:

### 1. Setup Environment
```bash
# Clone repository
git clone https://github.com/NagusameCS/Veritas.git
cd Veritas/training

# Install dependencies
pip install -r requirements.txt
```

### 2. Start Training in Background
```bash
# Start with nohup (training continues after logout)
./run_training.sh start

# Or use screen/tmux for interactive monitoring
screen -S training
./run_training.sh start
# Press Ctrl+A, D to detach
```

### 3. Monitor Remotely
```bash
# Check status
./run_training.sh status

# View logs
./run_training.sh logs

# Monitor in real-time
./run_training.sh monitor
```

### 4. Retrieve Results
After training completes:
```bash
# Check completion
./run_training.sh check

# Download model files
cd models
# Find the latest model directory
ls -lt

# Download using scp, rsync, or cloud storage
```

## Best Practices

1. **Use Quick Mode for Testing**
   - Always test with `--quick` first
   - Verify the process works before full training

2. **Monitor Resource Usage**
   - Training uses significant CPU and memory
   - Ensure adequate resources are available

3. **Save Training Logs**
   - Keep `sunrise_log.txt` for troubleshooting
   - Training receipts include verification hashes

4. **Backup Models**
   - Copy model files to safe storage
   - Training receipts enable reproducibility

5. **Use Version Control**
   - Tag successful model versions
   - Document training parameters used

## Automation Scripts

### Auto-restart on Failure
```bash
#!/bin/bash
while true; do
    ./run_training.sh start
    ./run_training.sh monitor &
    MONITOR_PID=$!
    
    # Wait for training to complete
    while ./run_training.sh status | grep -q "RUNNING"; do
        sleep 60
    done
    
    kill $MONITOR_PID 2>/dev/null
    
    # Check if successful
    if ./run_training.sh check; then
        echo "Training completed successfully!"
        break
    else
        echo "Training failed, restarting in 60 seconds..."
        sleep 60
    fi
done
```

### Scheduled Training (Cron)
```bash
# Run training daily at 2 AM
0 2 * * * cd /path/to/Veritas/training && ./run_training.sh start --quick
```

## Support

For issues or questions:
- Check logs: `./run_training.sh logs`
- Review error messages in `sunrise_log.txt`
- Ensure all dependencies are installed
- Verify network connectivity for dataset downloads

## Summary of Commands

```bash
# Start training
./run_training.sh start [args]

# Monitor in real-time
./run_training.sh monitor

# Check status
./run_training.sh status

# Check completion
./run_training.sh check

# Stop training
./run_training.sh stop

# View logs
./run_training.sh logs

# Get help
./run_training.sh help
```
