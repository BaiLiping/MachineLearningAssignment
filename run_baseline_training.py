#!/usr/bin/env python3
"""
Script to run baseline model training over 20 epochs and save results
"""

import torch
import torch.nn as nn
import numpy as np
import h5py
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, random_split, DataLoader
from tqdm import trange, tqdm
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
import json
import warnings
warnings.filterwarnings('ignore')

# Define the baseline model
class ModelBaseline(nn.Module):
    def __init__(self,):
        super(ModelBaseline, self).__init__()
        self.kernel_size = 3

        # conv layer
        downsample = self._downsample(4096, 128)
        self.conv1 = nn.Conv1d(in_channels=8, 
                               out_channels=32, 
                               kernel_size=self.kernel_size, 
                               stride=downsample,
                               padding=self._padding(downsample),
                               bias=False)
        
        # linear layer
        self.lin = nn.Linear(in_features=32*128,
                             out_features=1)
        
        # ReLU
        self.relu = nn.ReLU()

    def _padding(self, downsample):
        return max(0, int(np.floor((self.kernel_size - downsample + 1) / 2)))

    def _downsample(self, seq_len_in, seq_len_out):
        return int(seq_len_in // seq_len_out)

    def forward(self, x):
        x = x.transpose(2,1)
        x = self.relu(self.conv1(x))
        x_flat = x.view(x.size(0), -1)
        x = self.lin(x_flat)
        return x

# Set device - Use MPS for Mac Silicon
if torch.backends.mps.is_available():
    device = torch.device('mps')
    print(f"Using Apple Metal Performance Shaders (MPS) for acceleration")
elif torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"Using CUDA GPU for acceleration")
else:
    device = torch.device('cpu')
    print(f"Using CPU (no acceleration available)")

# Set random seed for reproducibility
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

# Training hyperparameters for baseline
baseline_learning_rate = 1e-2
baseline_weight_decay = 1e-1
baseline_num_epochs = 20
baseline_batch_size = 32

print(f"\nBaseline Model Hyperparameters:")
print(f"Learning Rate: {baseline_learning_rate}")
print(f"Weight Decay: {baseline_weight_decay}")
print(f"Epochs: {baseline_num_epochs}")
print(f"Batch Size: {baseline_batch_size}")

# Load and prepare data
print("\nLoading data...")
path_to_h5_train = 'codesubset/train.h5'
path_to_csv_train = 'codesubset/train.csv'
path_to_records = 'codesubset/train/RECORDS.txt'

# Load traces
with h5py.File(path_to_h5_train, 'r') as f:
    traces = torch.tensor(f['tracings'][()], dtype=torch.float32)

# Load labels
ids_traces = [int(x.split('TNMG')[1]) for x in list(pd.read_csv(path_to_records, header=None)[0])]
df = pd.read_csv(path_to_csv_train)
df.set_index('id_exam', inplace=True)
df = df.reindex(ids_traces)
labels = torch.tensor(np.array(df['AF']), dtype=torch.float32).reshape(-1,1)

# Create dataset
dataset = TensorDataset(traces, labels)
len_dataset = len(dataset)

# Split data (80-20)
train_size = int(0.8 * len_dataset)
valid_size = len_dataset - train_size
dataset_train, dataset_valid = random_split(dataset, [train_size, valid_size], 
                                          generator=torch.Generator().manual_seed(42))

# Create dataloaders
train_dataloader = DataLoader(dataset_train, batch_size=baseline_batch_size, shuffle=True)
valid_dataloader = DataLoader(dataset_valid, batch_size=baseline_batch_size, shuffle=False)

print(f"Training samples: {train_size}")
print(f"Validation samples: {valid_size}")

# Initialize baseline model
baseline_model = ModelBaseline()
baseline_model.to(device=device)

# Define loss function (standard BCE without weighting for baseline)
baseline_loss_function = nn.BCEWithLogitsLoss()

# Define optimizer
baseline_optimizer = torch.optim.Adam(baseline_model.parameters(), 
                                     lr=baseline_learning_rate, 
                                     weight_decay=baseline_weight_decay)

# Training history
baseline_train_loss = []
baseline_valid_loss = []
baseline_valid_auroc = []
baseline_valid_f1 = []
baseline_valid_ap = []

print("\nStarting baseline model training for 20 epochs...")
print("-" * 70)

# Training loop
for epoch in range(1, baseline_num_epochs + 1):
    # Training phase
    baseline_model.train()
    train_loss = 0
    n_train = 0
    
    train_pbar = tqdm(train_dataloader, desc=f"Epoch {epoch:2d} - Training")
    for traces_batch, labels_batch in train_pbar:
        traces_batch = traces_batch.to(device)
        labels_batch = labels_batch.to(device)
        
        baseline_optimizer.zero_grad()
        outputs = baseline_model(traces_batch)
        loss = baseline_loss_function(outputs, labels_batch)
        loss.backward()
        baseline_optimizer.step()
        
        train_loss += loss.item() * len(traces_batch)
        n_train += len(traces_batch)
        train_pbar.set_postfix({'loss': train_loss / n_train})
    
    avg_train_loss = train_loss / n_train
    baseline_train_loss.append(avg_train_loss)
    
    # Validation phase
    baseline_model.eval()
    valid_loss = 0
    n_valid = 0
    valid_preds = []
    valid_labels = []
    
    valid_pbar = tqdm(valid_dataloader, desc=f"Epoch {epoch:2d} - Validation")
    with torch.no_grad():
        for traces_batch, labels_batch in valid_pbar:
            traces_batch = traces_batch.to(device)
            labels_batch = labels_batch.to(device)
            
            outputs = baseline_model(traces_batch)
            loss = baseline_loss_function(outputs, labels_batch)
            
            valid_loss += loss.item() * len(traces_batch)
            n_valid += len(traces_batch)
            
            # Store predictions
            probs = torch.sigmoid(outputs)
            valid_preds.extend(probs.cpu().numpy())
            valid_labels.extend(labels_batch.cpu().numpy())
            
            valid_pbar.set_postfix({'loss': valid_loss / n_valid})
    
    avg_valid_loss = valid_loss / n_valid
    baseline_valid_loss.append(avg_valid_loss)
    
    # Calculate metrics
    valid_preds = np.array(valid_preds).flatten()
    valid_labels = np.array(valid_labels).flatten()
    
    auroc = roc_auc_score(valid_labels, valid_preds)
    ap = average_precision_score(valid_labels, valid_preds)
    f1 = f1_score(valid_labels, (valid_preds > 0.5).astype(int))
    
    baseline_valid_auroc.append(auroc)
    baseline_valid_ap.append(ap)
    baseline_valid_f1.append(f1)
    
    # Print epoch summary
    print(f"Epoch {epoch:2d}: Train Loss: {avg_train_loss:.4f} | "
          f"Valid Loss: {avg_valid_loss:.4f} | "
          f"AUROC: {auroc:.4f} | F1: {f1:.4f} | AP: {ap:.4f}")

print("-" * 70)
print("Baseline training completed!")

# Save results to JSON
results = {
    'device': str(device),
    'hyperparameters': {
        'learning_rate': baseline_learning_rate,
        'weight_decay': baseline_weight_decay,
        'num_epochs': baseline_num_epochs,
        'batch_size': baseline_batch_size
    },
    'train_loss': baseline_train_loss,
    'valid_loss': baseline_valid_loss,
    'valid_auroc': baseline_valid_auroc,
    'valid_f1': baseline_valid_f1,
    'valid_ap': baseline_valid_ap,
    'final_metrics': {
        'train_loss': baseline_train_loss[-1],
        'valid_loss': baseline_valid_loss[-1],
        'auroc': baseline_valid_auroc[-1],
        'f1': baseline_valid_f1[-1],
        'ap': baseline_valid_ap[-1]
    },
    'best_metrics': {
        'auroc': {'value': max(baseline_valid_auroc), 'epoch': baseline_valid_auroc.index(max(baseline_valid_auroc))+1},
        'f1': {'value': max(baseline_valid_f1), 'epoch': baseline_valid_f1.index(max(baseline_valid_f1))+1},
        'ap': {'value': max(baseline_valid_ap), 'epoch': baseline_valid_ap.index(max(baseline_valid_ap))+1}
    }
}

with open('baseline_training_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Plot results
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Loss curves
axes[0,0].plot(range(1, baseline_num_epochs+1), baseline_train_loss, 'b-', label='Train Loss', linewidth=2)
axes[0,0].plot(range(1, baseline_num_epochs+1), baseline_valid_loss, 'r-', label='Valid Loss', linewidth=2)
axes[0,0].set_xlabel('Epoch', fontsize=12)
axes[0,0].set_ylabel('Loss', fontsize=12)
axes[0,0].set_title('Baseline Model: Training and Validation Loss', fontsize=14)
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# AUROC curve
axes[0,1].plot(range(1, baseline_num_epochs+1), baseline_valid_auroc, 'orange', linewidth=2, marker='o', markersize=4)
axes[0,1].set_xlabel('Epoch', fontsize=12)
axes[0,1].set_ylabel('AUROC', fontsize=12)
axes[0,1].set_title('Baseline Model: Validation AUROC', fontsize=14)
axes[0,1].grid(True, alpha=0.3)
axes[0,1].set_ylim([0.5, 1.0])

# F1 Score curve
axes[0,2].plot(range(1, baseline_num_epochs+1), baseline_valid_f1, 'green', linewidth=2, marker='s', markersize=4)
axes[0,2].set_xlabel('Epoch', fontsize=12)
axes[0,2].set_ylabel('F1 Score', fontsize=12)
axes[0,2].set_title('Baseline Model: Validation F1 Score', fontsize=14)
axes[0,2].grid(True, alpha=0.3)
axes[0,2].set_ylim([0.0, 1.0])

# Average Precision curve
axes[1,0].plot(range(1, baseline_num_epochs+1), baseline_valid_ap, 'red', linewidth=2, marker='^', markersize=4)
axes[1,0].set_xlabel('Epoch', fontsize=12)
axes[1,0].set_ylabel('Average Precision', fontsize=12)
axes[1,0].set_title('Baseline Model: Validation Average Precision', fontsize=14)
axes[1,0].grid(True, alpha=0.3)
axes[1,0].set_ylim([0.5, 1.0])

# Combined metrics
axes[1,1].plot(range(1, baseline_num_epochs+1), baseline_valid_auroc, label='AUROC', linewidth=2)
axes[1,1].plot(range(1, baseline_num_epochs+1), baseline_valid_f1, label='F1', linewidth=2)
axes[1,1].plot(range(1, baseline_num_epochs+1), baseline_valid_ap, label='AP', linewidth=2)
axes[1,1].set_xlabel('Epoch', fontsize=12)
axes[1,1].set_ylabel('Score', fontsize=12)
axes[1,1].set_title('Baseline Model: All Validation Metrics', fontsize=14)
axes[1,1].legend()
axes[1,1].grid(True, alpha=0.3)
axes[1,1].set_ylim([0.5, 1.0])

# Training progress bar chart
epochs_to_show = [1, 5, 10, 15, 20]
metrics_at_epochs = {
    'AUROC': [baseline_valid_auroc[e-1] for e in epochs_to_show],
    'F1': [baseline_valid_f1[e-1] for e in epochs_to_show],
    'AP': [baseline_valid_ap[e-1] for e in epochs_to_show]
}

x = np.arange(len(epochs_to_show))
width = 0.25

for i, (metric, values) in enumerate(metrics_at_epochs.items()):
    axes[1,2].bar(x + i*width, values, width, label=metric)

axes[1,2].set_xlabel('Epoch', fontsize=12)
axes[1,2].set_ylabel('Score', fontsize=12)
axes[1,2].set_title('Baseline Model: Metrics at Key Epochs', fontsize=14)
axes[1,2].set_xticks(x + width)
axes[1,2].set_xticklabels(epochs_to_show)
axes[1,2].legend()
axes[1,2].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('baseline_training_results.png', dpi=150, bbox_inches='tight')
plt.show()

# Print final summary
print("\n" + "="*70)
print("BASELINE MODEL PERFORMANCE SUMMARY (20 Epochs)")
print("="*70)
print(f"Final Training Loss: {baseline_train_loss[-1]:.4f}")
print(f"Final Validation Loss: {baseline_valid_loss[-1]:.4f}")
print(f"Final AUROC: {baseline_valid_auroc[-1]:.4f}")
print(f"Final F1 Score: {baseline_valid_f1[-1]:.4f}")
print(f"Final Average Precision: {baseline_valid_ap[-1]:.4f}")
print("-"*70)
print(f"Best AUROC: {max(baseline_valid_auroc):.4f} (Epoch {baseline_valid_auroc.index(max(baseline_valid_auroc))+1})")
print(f"Best F1 Score: {max(baseline_valid_f1):.4f} (Epoch {baseline_valid_f1.index(max(baseline_valid_f1))+1})")
print(f"Best Average Precision: {max(baseline_valid_ap):.4f} (Epoch {baseline_valid_ap.index(max(baseline_valid_ap))+1})")
print("="*70)

# Check for overfitting
if baseline_valid_loss[-1] > min(baseline_valid_loss):
    print(f"\n⚠️  Warning: Model shows signs of overfitting.")
    print(f"   Best validation loss was {min(baseline_valid_loss):.4f} at epoch {baseline_valid_loss.index(min(baseline_valid_loss))+1}")
    print(f"   Final validation loss is {baseline_valid_loss[-1]:.4f}")
else:
    print(f"\n✓ Model continues to improve at epoch 20")

print(f"\n💻 Hardware acceleration: {'MPS (Apple Silicon)' if device.type == 'mps' else device.type.upper()}")
print(f"\n📊 Results saved to: baseline_training_results.json")
print(f"📈 Plots saved to: baseline_training_results.png")