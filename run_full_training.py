#!/usr/bin/env python3
"""Run the full notebook training"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, random_split, DataLoader
import h5py
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
import matplotlib.pyplot as plt

print("=" * 60)
print("RUNNING FULL ECG CLASSIFICATION TRAINING")
print("=" * 60)

# Set seed for reproducibility
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

# Hyperparameters
learning_rate = 1e-3
weight_decay = 1e-4
num_epochs = 25
batch_size = 16

# Device configuration - Use MPS for Mac Silicon
if torch.backends.mps.is_available():
    device = torch.device('mps')
    print(f"\nUsing device: MPS (Apple Silicon GPU)")
elif torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"\nUsing device: CUDA")
else:
    device = torch.device('cpu')
    print(f"\nUsing device: CPU")

# =============== Define Model ============================================#
print("\nDefining model...")

class Model(nn.Module):
    def __init__(self):
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
model.to(device)
print("✓ Model defined and moved to device")

# =============== Load Data ===============================================#
print("\nLoading data...")

# Load training data
path_to_h5_train = 'codesubset/train.h5'
path_to_csv_train = 'codesubset/train.csv'
path_to_records = 'codesubset/train/RECORDS.txt'

# Load traces
traces = torch.tensor(h5py.File(path_to_h5_train, 'r')['tracings'][()], dtype=torch.float32)

# Load labels
ids_traces = [int(x.split('TNMG')[1]) for x in list(pd.read_csv(path_to_records, header=None)[0])]
df = pd.read_csv(path_to_csv_train)
df.set_index('id_exam', inplace=True)
df = df.reindex(ids_traces)
labels = torch.tensor(np.array(df['AF']), dtype=torch.float32).reshape(-1,1)

# Create dataset
dataset = TensorDataset(traces, labels)
len_dataset = len(dataset)

print(f"✓ Loaded {len_dataset} samples")
print(f"  Class balance - AF: {torch.sum(labels).item():.0f} ({torch.sum(labels).item()/len_dataset*100:.1f}%)")

# Split data
train_size = int(0.8 * len_dataset)
valid_size = len_dataset - train_size
dataset_train, dataset_valid = random_split(dataset, [train_size, valid_size], 
                                          generator=torch.Generator().manual_seed(42))

# Create dataloaders
train_dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(dataset_valid, batch_size=batch_size, shuffle=False)

print(f"✓ Train set: {train_size} samples")
print(f"✓ Valid set: {valid_size} samples")

# =============== Define Loss and Optimizer ==============================#
print("\nSetting up training...")

# Weighted BCE loss for class imbalance
pos_weight = torch.tensor([len_dataset / (2 * torch.sum(labels))], device=device)
loss_function = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
print(f"✓ Loss function with pos_weight: {pos_weight.item():.2f}")

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                          factor=0.5, patience=3)
print("✓ Optimizer and scheduler configured")

# =============== Training Functions ======================================#

def train_loop(epoch, dataloader, model, optimizer, loss_function, device):
    model.train()
    total_loss = 0
    n_entries = 0
    
    train_pbar = tqdm(dataloader, desc=f"Training Epoch {epoch:2d}", leave=False)
    for traces, diagnoses in train_pbar:
        traces, diagnoses = traces.to(device), diagnoses.to(device)
        
        optimizer.zero_grad()
        outputs = model(traces)
        loss = loss_function(outputs, diagnoses)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.detach().cpu().numpy() * len(traces)
        n_entries += len(traces)
        
        train_pbar.set_postfix({'loss': total_loss / n_entries})
    
    return total_loss / n_entries

def eval_loop(epoch, dataloader, model, loss_function, device):
    model.eval()
    total_loss = 0
    n_entries = 0
    valid_probs = []
    valid_true = []
    
    eval_pbar = tqdm(dataloader, desc=f"Validation Epoch {epoch:2d}", leave=False)
    for traces_cpu, diagnoses_cpu in eval_pbar:
        traces, diagnoses = traces_cpu.to(device), diagnoses_cpu.to(device)
        
        with torch.no_grad():
            outputs = model(traces)
            loss = loss_function(outputs, diagnoses)
            probs = torch.sigmoid(outputs)
            valid_probs.append(probs.cpu().numpy())
            valid_true.append(diagnoses_cpu.numpy())
        
        total_loss += loss.detach().cpu().numpy() * len(traces)
        n_entries += len(traces)
        
        eval_pbar.set_postfix({'loss': total_loss / n_entries})
    
    return total_loss / n_entries, np.vstack(valid_probs), np.vstack(valid_true)

# =============== Training Loop ===========================================#
print("\n" + "=" * 60)
print("STARTING TRAINING")
print("=" * 60)

best_loss = np.inf
train_loss_all, valid_loss_all = [], []
valid_auroc_all, valid_f1_all, valid_ap_all = [], [], []

for epoch in range(1, num_epochs + 1):
    # Training
    train_loss = train_loop(epoch, train_dataloader, model, optimizer, loss_function, device)
    
    # Validation
    valid_loss, y_pred, y_true = eval_loop(epoch, valid_dataloader, model, loss_function, device)
    
    # Compute metrics
    y_pred_flat = y_pred.flatten()
    y_true_flat = y_true.flatten()
    valid_auroc = roc_auc_score(y_true_flat, y_pred_flat)
    valid_ap = average_precision_score(y_true_flat, y_pred_flat)
    y_pred_binary = (y_pred_flat > 0.5).astype(int)
    valid_f1 = f1_score(y_true_flat, y_pred_binary)
    
    # Store metrics
    train_loss_all.append(train_loss)
    valid_loss_all.append(valid_loss)
    valid_auroc_all.append(valid_auroc)
    valid_ap_all.append(valid_ap)
    valid_f1_all.append(valid_f1)
    
    # Save best model
    if valid_loss < best_loss:
        torch.save({'model': model.state_dict()}, 'model.pth')
        best_loss = valid_loss
        best_epoch = epoch
        model_save_state = " <- BEST"
    else:
        model_save_state = ""
    
    # Print progress
    print(f"Epoch {epoch:2d}/{num_epochs}: "
          f"Train Loss: {train_loss:.4f} | "
          f"Valid Loss: {valid_loss:.4f} | "
          f"AUROC: {valid_auroc:.4f} | "
          f"F1: {valid_f1:.4f} | "
          f"AP: {valid_ap:.4f}"
          f"{model_save_state}")
    
    # Update learning rate
    lr_scheduler.step(valid_loss)

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

# =============== Final Results ===========================================#
print(f"\nBest model saved from epoch {best_epoch}")
print(f"Best validation loss: {best_loss:.4f}")
print(f"\nFinal metrics (epoch {num_epochs}):")
print(f"  AUROC: {valid_auroc_all[-1]:.4f}")
print(f"  F1 Score: {valid_f1_all[-1]:.4f}")
print(f"  Average Precision: {valid_ap_all[-1]:.4f}")

# =============== Test Set Evaluation =====================================#
print("\n" + "=" * 60)
print("EVALUATING ON TEST SET")
print("=" * 60)

# Load best model
checkpoint = torch.load('model.pth', map_location=device)
model.load_state_dict(checkpoint['model'])
model.eval()

# Load test data
path_to_h5_test = 'codesubset/test.h5'
traces_test = torch.tensor(h5py.File(path_to_h5_test, 'r')['tracings'][()], dtype=torch.float32)
dataset_test = TensorDataset(traces_test)
test_dataloader = DataLoader(dataset_test, batch_size=32, shuffle=False)

# Make predictions
test_pred = []
for traces in tqdm(test_dataloader, desc="Testing"):
    traces = traces[0].to(device)
    with torch.no_grad():
        outputs = model(traces)
        probs = torch.sigmoid(outputs)
        test_pred.append(probs.cpu().numpy())

test_pred = np.vstack(test_pred)
soft_pred = np.hstack([1-test_pred, test_pred])

print(f"✓ Generated predictions for {len(test_pred)} test samples")
print(f"  Prediction shape: {soft_pred.shape}")
print(f"  Positive predictions: {np.sum(test_pred > 0.5):.0f} ({np.mean(test_pred > 0.5)*100:.1f}%)")

# Save predictions
np.save('test_predictions.npy', soft_pred)
print("✓ Predictions saved to test_predictions.npy")

# =============== Plot Results ============================================#
print("\nGenerating plots...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Loss curves
axes[0,0].plot(train_loss_all, label='Train Loss', color='blue')
axes[0,0].plot(valid_loss_all, label='Valid Loss', color='orange')
axes[0,0].axvline(x=best_epoch-1, color='red', linestyle='--', alpha=0.5, label=f'Best (epoch {best_epoch})')
axes[0,0].set_xlabel('Epoch')
axes[0,0].set_ylabel('Loss')
axes[0,0].set_title('Training and Validation Loss')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# AUROC curve
axes[0,1].plot(valid_auroc_all, color='green')
axes[0,1].axhline(y=0.97, color='red', linestyle='--', alpha=0.5, label='Target (0.97)')
axes[0,1].set_xlabel('Epoch')
axes[0,1].set_ylabel('AUROC')
axes[0,1].set_title('Validation AUROC')
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)

# F1 Score curve
axes[1,0].plot(valid_f1_all, color='purple')
axes[1,0].set_xlabel('Epoch')
axes[1,0].set_ylabel('F1 Score')
axes[1,0].set_title('Validation F1 Score')
axes[1,0].grid(True, alpha=0.3)

# Average Precision curve
axes[1,1].plot(valid_ap_all, color='brown')
axes[1,1].axhline(y=0.95, color='red', linestyle='--', alpha=0.5, label='Target (0.95)')
axes[1,1].set_xlabel('Epoch')
axes[1,1].set_ylabel('Average Precision')
axes[1,1].set_title('Validation Average Precision')
axes[1,1].legend()
axes[1,1].grid(True, alpha=0.3)

plt.suptitle('ECG Classification Training Results', fontsize=16)
plt.tight_layout()
plt.savefig('training_results.png', dpi=100, bbox_inches='tight')
print("✓ Plots saved to training_results.png")

print("\n" + "=" * 60)
print("✅ NOTEBOOK EXECUTION COMPLETE!")
print("=" * 60)
print("\nOutputs generated:")
print("  - model.pth (best model checkpoint)")
print("  - test_predictions.npy (test set predictions)")
print("  - training_results.png (performance plots)")