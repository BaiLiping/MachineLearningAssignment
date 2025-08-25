#!/usr/bin/env python3
"""
Script to run linear model training over 20 epochs with decimation
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

# Define the linear model with decimation
class Model(nn.Module):
    def __init__(self, ):
        super(Model, self).__init__()
        self.decimation_factor = 4
        self.flatten = nn.Flatten()
        self.network = nn.Sequential(
            nn.Linear(int(4096 / self.decimation_factor) * 8, 512), # 4096 samples, 8 leads
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            #nn.ReLU()
        )
    def forward(self, x):
        x = self.flatten(x[:, ::self.decimation_factor, :])
        logits = self.network(x)
        return logits

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

# Training hyperparameters
learning_rate = 1e-3
weight_decay = 1e-4
num_epochs = 20
batch_size = 32

print(f"\nLinear Model Hyperparameters:")
print(f"Learning Rate: {learning_rate}")
print(f"Weight Decay: {weight_decay}")
print(f"Epochs: {num_epochs}")
print(f"Batch Size: {batch_size}")
print(f"Decimation Factor: 4 (reduces 4096 samples to 1024)")

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

# Calculate class weights for balanced loss
pos_count = torch.sum(labels)
neg_count = len_dataset - pos_count
pos_weight = neg_count / pos_count
print(f"Class imbalance - Positive weight: {pos_weight:.2f}")

# Split data (80-20)
train_size = int(0.8 * len_dataset)
valid_size = len_dataset - train_size
dataset_train, dataset_valid = random_split(dataset, [train_size, valid_size], 
                                          generator=torch.Generator().manual_seed(42))

# Create dataloaders
train_dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(dataset_valid, batch_size=batch_size, shuffle=False)

print(f"Training samples: {train_size}")
print(f"Validation samples: {valid_size}")

# Initialize model
model = Model()
model.to(device=device)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total model parameters: {total_params:,}")

# Define weighted loss function
loss_function = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))

# Define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

# Training history
train_loss_history = []
valid_loss_history = []
valid_auroc_history = []
valid_f1_history = []
valid_ap_history = []

print("\nStarting linear model training for 20 epochs...")
print("-" * 70)

best_auroc = 0
best_epoch = 0

# Training loop
for epoch in range(1, num_epochs + 1):
    # Training phase
    model.train()
    train_loss = 0
    n_train = 0
    
    train_pbar = tqdm(train_dataloader, desc=f"Epoch {epoch:2d} - Training")
    for traces_batch, labels_batch in train_pbar:
        traces_batch = traces_batch.to(device)
        labels_batch = labels_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(traces_batch)
        loss = loss_function(outputs, labels_batch)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * len(traces_batch)
        n_train += len(traces_batch)
        train_pbar.set_postfix({'loss': train_loss / n_train})
    
    avg_train_loss = train_loss / n_train
    train_loss_history.append(avg_train_loss)
    
    # Validation phase
    model.eval()
    valid_loss = 0
    n_valid = 0
    valid_preds = []
    valid_labels = []
    
    valid_pbar = tqdm(valid_dataloader, desc=f"Epoch {epoch:2d} - Validation")
    with torch.no_grad():
        for traces_batch, labels_batch in valid_pbar:
            traces_batch = traces_batch.to(device)
            labels_batch = labels_batch.to(device)
            
            outputs = model(traces_batch)
            loss = loss_function(outputs, labels_batch)
            
            valid_loss += loss.item() * len(traces_batch)
            n_valid += len(traces_batch)
            
            # Store predictions
            probs = torch.sigmoid(outputs)
            valid_preds.extend(probs.cpu().numpy())
            valid_labels.extend(labels_batch.cpu().numpy())
            
            valid_pbar.set_postfix({'loss': valid_loss / n_valid})
    
    avg_valid_loss = valid_loss / n_valid
    valid_loss_history.append(avg_valid_loss)
    
    # Calculate metrics
    valid_preds = np.array(valid_preds).flatten()
    valid_labels = np.array(valid_labels).flatten()
    
    auroc = roc_auc_score(valid_labels, valid_preds)
    ap = average_precision_score(valid_labels, valid_preds)
    f1 = f1_score(valid_labels, (valid_preds > 0.5).astype(int))
    
    valid_auroc_history.append(auroc)
    valid_ap_history.append(ap)
    valid_f1_history.append(f1)
    
    # Save best model
    if auroc > best_auroc:
        best_auroc = auroc
        best_epoch = epoch
        torch.save({'model': model.state_dict(), 'epoch': epoch, 'auroc': auroc}, 'linear_model_best.pth')
    
    # Print epoch summary
    print(f"Epoch {epoch:2d}: Train Loss: {avg_train_loss:.4f} | "
          f"Valid Loss: {avg_valid_loss:.4f} | "
          f"AUROC: {auroc:.4f} | F1: {f1:.4f} | AP: {ap:.4f}")
    
    # Learning rate scheduling
    scheduler.step(avg_valid_loss)

print("-" * 70)
print("Linear model training completed!")

# Save results to JSON
results = {
    'device': str(device),
    'model_type': 'Linear with decimation',
    'hyperparameters': {
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'decimation_factor': 4,
        'total_parameters': total_params
    },
    'train_loss': train_loss_history,
    'valid_loss': valid_loss_history,
    'valid_auroc': valid_auroc_history,
    'valid_f1': valid_f1_history,
    'valid_ap': valid_ap_history,
    'final_metrics': {
        'train_loss': train_loss_history[-1],
        'valid_loss': valid_loss_history[-1],
        'auroc': valid_auroc_history[-1],
        'f1': valid_f1_history[-1],
        'ap': valid_ap_history[-1]
    },
    'best_metrics': {
        'auroc': {'value': max(valid_auroc_history), 'epoch': valid_auroc_history.index(max(valid_auroc_history))+1},
        'f1': {'value': max(valid_f1_history), 'epoch': valid_f1_history.index(max(valid_f1_history))+1},
        'ap': {'value': max(valid_ap_history), 'epoch': valid_ap_history.index(max(valid_ap_history))+1}
    }
}

with open('linear_model_training_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Plot results
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Loss curves
axes[0,0].plot(range(1, num_epochs+1), train_loss_history, 'b-', label='Train Loss', linewidth=2)
axes[0,0].plot(range(1, num_epochs+1), valid_loss_history, 'r-', label='Valid Loss', linewidth=2)
axes[0,0].set_xlabel('Epoch', fontsize=12)
axes[0,0].set_ylabel('Loss', fontsize=12)
axes[0,0].set_title('Linear Model: Training and Validation Loss', fontsize=14)
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# AUROC curve
axes[0,1].plot(range(1, num_epochs+1), valid_auroc_history, 'orange', linewidth=2, marker='o', markersize=4)
axes[0,1].set_xlabel('Epoch', fontsize=12)
axes[0,1].set_ylabel('AUROC', fontsize=12)
axes[0,1].set_title('Linear Model: Validation AUROC', fontsize=14)
axes[0,1].grid(True, alpha=0.3)
axes[0,1].set_ylim([0.5, 1.0])
axes[0,1].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

# F1 Score curve
axes[0,2].plot(range(1, num_epochs+1), valid_f1_history, 'green', linewidth=2, marker='s', markersize=4)
axes[0,2].set_xlabel('Epoch', fontsize=12)
axes[0,2].set_ylabel('F1 Score', fontsize=12)
axes[0,2].set_title('Linear Model: Validation F1 Score', fontsize=14)
axes[0,2].grid(True, alpha=0.3)
axes[0,2].set_ylim([0.0, 1.0])

# Average Precision curve
axes[1,0].plot(range(1, num_epochs+1), valid_ap_history, 'red', linewidth=2, marker='^', markersize=4)
axes[1,0].set_xlabel('Epoch', fontsize=12)
axes[1,0].set_ylabel('Average Precision', fontsize=12)
axes[1,0].set_title('Linear Model: Validation Average Precision', fontsize=14)
axes[1,0].grid(True, alpha=0.3)
axes[1,0].set_ylim([0.0, 1.0])

# Combined metrics
axes[1,1].plot(range(1, num_epochs+1), valid_auroc_history, label='AUROC', linewidth=2)
axes[1,1].plot(range(1, num_epochs+1), valid_f1_history, label='F1', linewidth=2)
axes[1,1].plot(range(1, num_epochs+1), valid_ap_history, label='AP', linewidth=2)
axes[1,1].set_xlabel('Epoch', fontsize=12)
axes[1,1].set_ylabel('Score', fontsize=12)
axes[1,1].set_title('Linear Model: All Validation Metrics', fontsize=14)
axes[1,1].legend()
axes[1,1].grid(True, alpha=0.3)
axes[1,1].set_ylim([0.0, 1.0])

# Comparison with baseline
baseline_auroc = [0.6432, 0.5145, 0.4787, 0.5247, 0.4597, 0.4481, 0.5464, 0.5618, 
                  0.5441, 0.5870, 0.5428, 0.4381, 0.5277, 0.4764, 0.5452, 0.5860, 
                  0.5148, 0.5690, 0.4504, 0.5093]

axes[1,2].plot(range(1, 21), baseline_auroc, 'gray', linewidth=1.5, alpha=0.5, label='Baseline Model')
axes[1,2].plot(range(1, num_epochs+1), valid_auroc_history, 'blue', linewidth=2, label='Linear Model')
axes[1,2].set_xlabel('Epoch', fontsize=12)
axes[1,2].set_ylabel('AUROC', fontsize=12)
axes[1,2].set_title('Model Comparison: AUROC', fontsize=14)
axes[1,2].legend()
axes[1,2].grid(True, alpha=0.3)
axes[1,2].set_ylim([0.3, 1.0])

plt.tight_layout()
plt.savefig('linear_model_training_results.png', dpi=150, bbox_inches='tight')
plt.show()

# Print final summary
print("\n" + "="*70)
print("LINEAR MODEL PERFORMANCE SUMMARY (20 Epochs)")
print("="*70)
print(f"Model Parameters: {total_params:,}")
print(f"Decimation Factor: 4 (4096 → 1024 samples)")
print("-"*70)
print(f"Final Training Loss: {train_loss_history[-1]:.4f}")
print(f"Final Validation Loss: {valid_loss_history[-1]:.4f}")
print(f"Final AUROC: {valid_auroc_history[-1]:.4f}")
print(f"Final F1 Score: {valid_f1_history[-1]:.4f}")
print(f"Final Average Precision: {valid_ap_history[-1]:.4f}")
print("-"*70)
print(f"Best AUROC: {max(valid_auroc_history):.4f} (Epoch {valid_auroc_history.index(max(valid_auroc_history))+1})")
print(f"Best F1 Score: {max(valid_f1_history):.4f} (Epoch {valid_f1_history.index(max(valid_f1_history))+1})")
print(f"Best Average Precision: {max(valid_ap_history):.4f} (Epoch {valid_ap_history.index(max(valid_ap_history))+1})")
print("="*70)

print(f"\n💻 Hardware acceleration: {'MPS (Apple Silicon)' if device.type == 'mps' else device.type.upper()}")
print(f"📊 Results saved to: linear_model_training_results.json")
print(f"📈 Plots saved to: linear_model_training_results.png")
print(f"🏆 Best model saved to: linear_model_best.pth (Epoch {best_epoch}, AUROC: {best_auroc:.4f})")