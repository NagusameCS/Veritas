#!/bin/bash
# Veritas Model Training Automation Script
# This script automates the training, monitoring, and management of the Sunrise ML model

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="${SCRIPT_DIR}/sunrise_log.txt"
PID_FILE="${SCRIPT_DIR}/.training.pid"
STATUS_FILE="${SCRIPT_DIR}/.training_status"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

print_error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1"
}

# Function to start training
start_training() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p "$PID" > /dev/null 2>&1; then
            print_error "Training is already running (PID: $PID)"
            return 1
        else
            print_warning "Stale PID file found, removing..."
            rm -f "$PID_FILE"
        fi
    fi
    
    print_status "Starting model training..."
    
    # Clear log file
    > "$LOG_FILE"
    
    # Start training in background (note: 'start' was already shifted off, so $@ contains only training args)
    cd "$SCRIPT_DIR"
    nohup python3 train_sunrise.py "$@" > "$LOG_FILE" 2>&1 &
    
    # Save PID
    echo $! > "$PID_FILE"
    echo "running" > "$STATUS_FILE"
    
    print_status "Training started with PID: $(cat $PID_FILE)"
    print_status "Logs: $LOG_FILE"
    print_status "Use './run_training.sh monitor' to watch progress"
    print_status "Use './run_training.sh status' to check status"
}

# Function to monitor training
monitor_training() {
    if [ ! -f "$LOG_FILE" ]; then
        print_error "Log file not found. Training may not have started."
        return 1
    fi
    
    print_status "Monitoring training logs (Ctrl+C to exit)..."
    tail -f "$LOG_FILE"
}

# Function to check status
check_status() {
    if [ ! -f "$PID_FILE" ]; then
        print_status "Training is not running"
        return 0
    fi
    
    PID=$(cat "$PID_FILE")
    if ps -p "$PID" > /dev/null 2>&1; then
        print_status "Training is RUNNING (PID: $PID)"
        
        # Check log file for recent activity
        if [ -f "$LOG_FILE" ]; then
            print_status "Recent log entries:"
            tail -n 10 "$LOG_FILE"
            
            # Try to extract accuracy if available
            ACCURACY=$(grep -i "accuracy" "$LOG_FILE" | tail -1 || echo "")
            if [ -n "$ACCURACY" ]; then
                echo ""
                print_status "Latest accuracy: $ACCURACY"
            fi
        fi
    else
        print_warning "Training process (PID: $PID) is not running"
        
        # Check if it completed successfully
        if [ -f "$LOG_FILE" ]; then
            if grep -q "TRAINING COMPLETE" "$LOG_FILE"; then
                print_status "Training completed successfully!"
                echo "completed" > "$STATUS_FILE"
                rm -f "$PID_FILE"
            elif grep -q "ERROR" "$LOG_FILE"; then
                print_error "Training failed with errors"
                echo "failed" > "$STATUS_FILE"
            else
                print_warning "Training stopped unexpectedly"
                echo "stopped" > "$STATUS_FILE"
            fi
        fi
    fi
}

# Function to stop training
stop_training() {
    if [ ! -f "$PID_FILE" ]; then
        print_error "No training process found"
        return 1
    fi
    
    PID=$(cat "$PID_FILE")
    if ps -p "$PID" > /dev/null 2>&1; then
        print_status "Stopping training (PID: $PID)..."
        kill "$PID"
        
        # Wait for process to stop (with timeout)
        local timeout=10
        local elapsed=0
        while ps -p "$PID" > /dev/null 2>&1 && [ $elapsed -lt $timeout ]; do
            sleep 1
            elapsed=$((elapsed + 1))
        done
        
        if ps -p "$PID" > /dev/null 2>&1; then
            print_warning "Process didn't stop gracefully, force killing..."
            kill -9 "$PID"
            sleep 1
        fi
        
        print_status "Training stopped"
        echo "stopped" > "$STATUS_FILE"
        rm -f "$PID_FILE"
    else
        print_warning "Training process (PID: $PID) is not running"
        rm -f "$PID_FILE"
    fi
}

# Function to show logs
show_logs() {
    if [ ! -f "$LOG_FILE" ]; then
        print_error "Log file not found"
        return 1
    fi
    
    cat "$LOG_FILE"
}

# Function to check completion
check_completion() {
    if [ ! -f "$LOG_FILE" ]; then
        print_error "Log file not found"
        return 1
    fi
    
    if grep -q "TRAINING COMPLETE" "$LOG_FILE"; then
        print_status "Training has COMPLETED successfully!"
        
        # Extract final results
        echo ""
        print_status "Training Results:"
        grep -A 20 "TRAINING RESULTS:" "$LOG_FILE" || echo "Results not found in log"
        
        return 0
    else
        print_status "Training is still in progress or has not completed successfully"
        return 1
    fi
}

# Function to show help
show_help() {
    cat << EOF
Veritas Model Training Automation

Usage: $0 <command> [options]

Commands:
    start [args]    Start model training (passes args to train_sunrise.py)
                    Examples:
                      $0 start
                      $0 start --quick
                      $0 start --trials 50 --max-samples-per-dataset 10000
    
    stop            Stop running training process
    
    status          Check training status and show recent logs
    
    monitor         Monitor training logs in real-time (tail -f)
    
    logs            Show full training logs
    
    check           Check if training has completed successfully
    
    help            Show this help message

Examples:
    # Start training with default settings
    $0 start
    
    # Start quick training (fewer samples and trials)
    $0 start --quick
    
    # Monitor training in another terminal
    $0 monitor
    
    # Check status periodically
    $0 status
    
    # Check if completed
    $0 check
    
    # Stop training if needed
    $0 stop

Files:
    Log file:    $LOG_FILE
    PID file:    $PID_FILE
    Status file: $STATUS_FILE

EOF
}

# Main command dispatcher
case "${1:-help}" in
    start)
        shift
        start_training "$@"
        ;;
    stop)
        stop_training
        ;;
    status)
        check_status
        ;;
    monitor)
        monitor_training
        ;;
    logs)
        show_logs
        ;;
    check)
        check_completion
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        print_error "Unknown command: $1"
        echo ""
        show_help
        exit 1
        ;;
esac
