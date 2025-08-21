#!/usr/bin/env python3
"""Check the status of the training"""

import os
import time

# Check if model file exists
if os.path.exists('model.pth'):
    mod_time = os.path.getmtime('model.pth')
    current_time = time.time()
    minutes_ago = (current_time - mod_time) / 60
    print(f"✓ Model checkpoint found (last updated {minutes_ago:.1f} minutes ago)")
else:
    print("✗ No model checkpoint yet")

# Check if predictions exist
if os.path.exists('test_predictions.npy'):
    print("✓ Test predictions generated")
else:
    print("⏳ Test predictions not yet generated")

# Check if plots exist
if os.path.exists('training_results.png'):
    print("✓ Training plots saved")
else:
    print("⏳ Training plots not yet generated")

# Check if training is still running
import subprocess
result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
if 'run_full_training.py' in result.stdout:
    print("\n🏃 Training is still running...")
else:
    print("\n✅ Training appears to be complete!")
    
print("\nTo see the latest training progress, check the console output.")
print("The full training of 25 epochs takes approximately 20-25 minutes.")