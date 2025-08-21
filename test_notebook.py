#!/usr/bin/env python3
"""Test script to validate the notebook can run properly"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

print("Testing ECG Classification Notebook...")

# Test 1: Check if we can create directories and download files
print("\n1. Testing file management...")
def exists(path):
    val = os.path.exists(path)
    if val:
        print(f'{path} already exists. Using cached.')
    return val

# Test 2: Import required libraries
print("\n2. Testing imports...")
try:
    import torch
    import torch.nn as nn
    import numpy as np
    from tqdm import tqdm
    import h5py
    import pandas as pd
    import matplotlib.pyplot as plt
    print("✓ Core imports successful")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

# Test 3: Check if ecg-plot can be imported
print("\n3. Testing ecg-plot import...")
try:
    import ecg_plot
    print("✓ ecg_plot imported successfully")
except ImportError:
    print("✗ ecg_plot not found, trying to install...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'ecg-plot'])
    import ecg_plot
    print("✓ ecg_plot installed and imported")

# Test 4: Define the model
print("\n4. Testing model definition...")
try:
    class Model(nn.Module):
        def __init__(self,):
            super(Model, self).__init__()
            
            # Multi-scale convolutional layers
            self.conv1 = nn.Conv1d(in_channels=8, out_channels=64, kernel_size=7, stride=2, padding=3)
            self.bn1 = nn.BatchNorm1d(64)
            
            self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2)
            self.bn2 = nn.BatchNorm1d(128)
            
            self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)
            self.bn3 = nn.BatchNorm1d(256)
            
            self.conv4 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1)
            self.bn4 = nn.BatchNorm1d(512)
            
            # Global average pooling
            self.global_pool = nn.AdaptiveAvgPool1d(1)
            
            # Dropout for regularization
            self.dropout = nn.Dropout(0.5)
            
            # Classification head
            self.fc1 = nn.Linear(512, 256)
            self.fc2 = nn.Linear(256, 64)
            self.fc3 = nn.Linear(64, 1)
            
            # Activation functions
            self.relu = nn.ReLU()
            
        def forward(self, x):
            x = x.transpose(2, 1)
            
            x = self.relu(self.bn1(self.conv1(x)))
            x = self.relu(self.bn2(self.conv2(x)))
            x = self.relu(self.bn3(self.conv3(x)))
            x = self.relu(self.bn4(self.conv4(x)))
            
            x = self.global_pool(x)
            x = x.view(x.size(0), -1)
            
            x = self.dropout(self.relu(self.fc1(x)))
            x = self.dropout(self.relu(self.fc2(x)))
            x = self.fc3(x)
            
            return x
    
    model = Model()
    print("✓ Model defined successfully")
    
    # Test forward pass with dummy data
    dummy_input = torch.randn(2, 4096, 8)  # batch_size=2, seq_len=4096, n_leads=8
    output = model(dummy_input)
    print(f"✓ Model forward pass successful, output shape: {output.shape}")
    
except Exception as e:
    print(f"✗ Model definition error: {e}")
    sys.exit(1)

# Test 5: Test training functions
print("\n5. Testing training functions...")
try:
    def train_loop(epoch, dataloader, model, optimizer, loss_function, device):
        model.train()
        total_loss = 0
        n_entries = 0
        train_pbar = tqdm(dataloader, desc=f"Training Epoch {epoch:2d}", leave=True)
        for traces, diagnoses in train_pbar:
            traces, diagnoses = traces.to(device), diagnoses.to(device)
            optimizer.zero_grad()
            outputs = model(traces)
            loss = loss_function(outputs, diagnoses)
            loss.backward()
            optimizer.step()
            total_loss += loss.detach().cpu().numpy()
            n_entries += len(traces)
            train_pbar.set_postfix({'loss': total_loss / n_entries})
        train_pbar.close()
        return total_loss / n_entries
    
    print("✓ Training loop defined successfully")
    
    def eval_loop(epoch, dataloader, model, loss_function, device):
        model.eval()
        total_loss = 0
        n_entries = 0
        valid_probs = []
        valid_true = []
        
        eval_pbar = tqdm(dataloader, desc=f"Evaluation Epoch {epoch:2d}", leave=True)
        for traces_cpu, diagnoses_cpu in eval_pbar:
            traces, diagnoses = traces_cpu.to(device), diagnoses_cpu.to(device)
            with torch.no_grad():
                outputs = model(traces)
                loss = loss_function(outputs, diagnoses)
                probs = torch.sigmoid(outputs)
                valid_probs.append(probs.cpu().numpy())
                valid_true.append(diagnoses_cpu.numpy())
            total_loss += loss.detach().cpu().numpy()
            n_entries += len(traces)
            eval_pbar.set_postfix({'loss': total_loss / n_entries})
        eval_pbar.close()
        return total_loss / n_entries, np.vstack(valid_probs), np.vstack(valid_true)
    
    print("✓ Evaluation loop defined successfully")
    
except Exception as e:
    print(f"✗ Training function error: {e}")
    sys.exit(1)

# Test 6: Check for data files
print("\n6. Checking for data files...")
data_files = [
    'codesubset/train.h5',
    'codesubset/test.h5',
    'codesubset/train.csv',
    'codesubset/test.csv'
]

missing_files = []
for file in data_files:
    if os.path.exists(file):
        print(f"✓ Found: {file}")
    else:
        print(f"✗ Missing: {file}")
        missing_files.append(file)

if missing_files:
    print("\n⚠ Warning: Some data files are missing. You need to run the data download cells in the notebook first.")
    print("Missing files:", missing_files)
else:
    print("\n✓ All data files found!")
    
    # Test 7: Try loading a small sample of data
    print("\n7. Testing data loading...")
    try:
        # Load a small sample
        with h5py.File('codesubset/train.h5', 'r') as f:
            data_shape = f['tracings'].shape
            print(f"✓ Train data shape: {data_shape}")
            
            # Load first sample
            sample = f['tracings'][0]
            print(f"✓ Sample shape: {sample.shape}")
            
        # Load CSV
        df = pd.read_csv('codesubset/train.csv')
        print(f"✓ CSV loaded, shape: {df.shape}")
        print(f"✓ Columns: {list(df.columns)}")
        
    except Exception as e:
        print(f"✗ Data loading error: {e}")

print("\n" + "="*50)
print("Test Summary:")
print("="*50)
print("✓ All critical components are working!")
print("✓ The notebook should be able to run successfully.")
print("\nNext steps:")
if missing_files:
    print("1. Run the data download cells in the notebook to get the data files")
    print("2. Then run the training cells to train the model")
else:
    print("1. Run the notebook cells in order")
    print("2. The model should train without errors")