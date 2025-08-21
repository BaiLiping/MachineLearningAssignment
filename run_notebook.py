#!/usr/bin/env python3
"""Run the notebook cells programmatically"""

import os
import sys
import subprocess
import warnings
warnings.filterwarnings('ignore')

print("Running ECG Classification Notebook...")
print("="*50)

# Helper function
def exists(path):
    val = os.path.exists(path)
    if val:
        print(f'{path} already exists. Using cached.')
    return val

# Cell 4: Clone requirements if needed
print("\n[Cell 4] Checking requirements.txt...")
if not exists('requirements.txt'):
    subprocess.run(['git', 'clone', 'https://gist.github.com/dgedon/8a7b91714568dc35d0527233e9ceada4.git', 'req'])
    subprocess.run(['mv', 'req/requirements.txt', '.'])
    subprocess.run(['rm', '-rf', 'req'])

# Cell 5: Install packages - already done in venv
print("\n[Cell 5] Packages already installed in venv")

# Cell 6: Import libraries
print("\n[Cell 6] Importing libraries...")
import torch
import torch.nn as nn
import numpy as np
from tqdm.notebook import trange, tqdm
import h5py
import pandas as pd
import matplotlib.pyplot as plt
print("✓ Libraries imported")

# Cell 8: Download dataset
print("\n[Cell 8] Downloading dataset...")
if not exists('codesubset.tar.gz'):
    print("Downloading dataset from Dropbox...")
    subprocess.run(['wget', 'https://www.dropbox.com/s/9zkqa5y5jqakdil/codesubset.tar.gz?dl=0', '-O', 'codesubset.tar.gz'], check=True)

# Cell 9: Unzip dataset
print("\n[Cell 9] Extracting dataset...")
if not exists('codesubset'):
    subprocess.run(['tar', '-xf', 'codesubset.tar.gz'], check=True)
    print("✓ Dataset extracted")

# Cell 12: Clone preprocessing code
print("\n[Cell 12] Cloning preprocessing code...")
if not exists('ecg-preprocessing'):
    subprocess.run(['git', 'clone', 'https://github.com/paulhausner/ecg-preprocessing.git'], check=True)

# Cell 14: Test ECG plotting
print("\n[Cell 14] Testing ECG plotting...")
sys.path.append('ecg-preprocessing')
from read_ecg import read_ecg

try:
    import ecg_plot
except ImportError:
    subprocess.check_call(['pip', 'install', 'ecg-plot'])
    import ecg_plot

PATH_TO_WFDB = 'codesubset/train/TNMG100046'
ecg_sample, sample_rate, _ = read_ecg(PATH_TO_WFDB)
print(f"✓ ECG sample loaded, shape: {ecg_sample.shape}, sample rate: {sample_rate}")

# Cell 18: Generate preprocessed h5 files
print("\n[Cell 18] Generating preprocessed h5 files...")
if not exists('codesubset/train.h5'):
    print("Preprocessing training data...")
    subprocess.run([
        'python', 'ecg-preprocessing/generate_h5.py',
        '--new_freq', '400', '--new_len', '4096',
        '--remove_baseline', '--powerline', '60',
        'codesubset/train/RECORDS.txt', 'codesubset/train.h5'
    ], check=True)

if not exists('codesubset/test.h5'):
    print("Preprocessing test data...")
    subprocess.run([
        'python', 'ecg-preprocessing/generate_h5.py',
        '--new_freq', '400', '--new_len', '4096',
        '--remove_baseline', '--powerline', '60',
        'codesubset/test/RECORDS.txt', 'codesubset/test.h5'
    ], check=True)

print("✓ Preprocessed h5 files ready")

# Cell 20: Data Analysis
print("\n[Cell 20] Performing data analysis...")
PATH_TO_H5_FILE = 'codesubset/train.h5'
f = h5py.File(PATH_TO_H5_FILE, 'r')
data = f['tracings']

path_to_csv_train = 'codesubset/train.csv'
path_to_records = 'codesubset/train/RECORDS.txt'
ids_traces = [int(x.split('TNMG')[1]) for x in list(pd.read_csv(path_to_records, header=None)[0])]
df = pd.read_csv(path_to_csv_train)
df.set_index('id_exam', inplace=True)
df = df.reindex(ids_traces)

print(f"Dataset shape: {data.shape}")
print(f"Number of samples: {data.shape[0]}")
print(f"Sequence length: {data.shape[1]}")
print(f"Number of leads: {data.shape[2]}")

af_labels = df['AF'].values
print(f"\nClass balance:")
print(f"AF cases: {np.sum(af_labels)} ({np.mean(af_labels)*100:.1f}%)")
print(f"Non-AF cases: {np.sum(1-af_labels)} ({(1-np.mean(af_labels))*100:.1f}%)")

f.close()

# Cell 23 & 25: Define models
print("\n[Cell 23 & 25] Defining models...")

class ModelBaseline(nn.Module):
    def __init__(self,):
        super(ModelBaseline, self).__init__()
        self.kernel_size = 3
        downsample = self._downsample(4096, 128)
        self.conv1 = nn.Conv1d(in_channels=8, out_channels=32, 
                               kernel_size=self.kernel_size, 
                               stride=downsample,
                               padding=self._padding(downsample),
                               bias=False)
        self.lin = nn.Linear(in_features=32*128, out_features=1)
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

class Model(nn.Module):
    def __init__(self,):
        super(Model, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=8, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.conv4 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm1d(512)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 1)
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

print("✓ Models defined")

# Cell 29 & 32: Define training and evaluation loops
print("\n[Cell 29 & 32] Defining training functions...")

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

print("✓ Training functions defined")

# Run a quick training test with 1 epoch to verify everything works
print("\n" + "="*50)
print("Running quick training test (1 epoch)...")
print("="*50)

from torch.utils.data import TensorDataset, random_split, DataLoader
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score

# Set parameters
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
learning_rate = 1e-3
weight_decay = 1e-4
num_epochs = 1  # Just 1 epoch for testing
batch_size = 16

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load data
print("\nLoading data...")
traces = torch.tensor(h5py.File('codesubset/train.h5', 'r')['tracings'][()], dtype=torch.float32)
labels = torch.tensor(np.array(df['AF']), dtype=torch.float32).reshape(-1,1)
dataset = TensorDataset(traces, labels)
len_dataset = len(dataset)

# Split data
train_size = int(0.8 * len_dataset)
valid_size = len_dataset - train_size
dataset_train, dataset_valid = random_split(dataset, [train_size, valid_size], 
                                          generator=torch.Generator().manual_seed(42))

# Create dataloaders
train_dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(dataset_valid, batch_size=batch_size, shuffle=False)

# Initialize model
model = Model()
model.to(device=device)

# Define loss and optimizer
pos_weight = torch.tensor([len_dataset / (2 * torch.sum(labels))], device=device)
loss_function = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Train for 1 epoch
print("\nTraining for 1 epoch...")
train_loss = train_loop(1, train_dataloader, model, optimizer, loss_function, device)
valid_loss, y_pred, y_true = eval_loop(1, valid_dataloader, model, loss_function, device)

# Calculate metrics
y_pred_flat = y_pred.flatten()
y_true_flat = y_true.flatten()
valid_auroc = roc_auc_score(y_true_flat, y_pred_flat)
valid_ap = average_precision_score(y_true_flat, y_pred_flat)
y_pred_binary = (y_pred_flat > 0.5).astype(int)
valid_f1 = f1_score(y_true_flat, y_pred_binary)

print(f"\nResults after 1 epoch:")
print(f"Train Loss: {train_loss:.6f}")
print(f"Valid Loss: {valid_loss:.6f}")
print(f"AUROC: {valid_auroc:.4f}")
print(f"F1 Score: {valid_f1:.4f}")
print(f"Average Precision: {valid_ap:.4f}")

# Save model
torch.save({'model': model.state_dict()}, 'model_test.pth')
print("\n✓ Model saved to model_test.pth")

print("\n" + "="*50)
print("SUCCESS! Notebook runs without errors!")
print("="*50)
print("\nThe notebook is fully functional and can:")
print("1. Download and preprocess data")
print("2. Define and initialize the model")
print("3. Train and evaluate the model")
print("4. Save model checkpoints")
print("\nTo train the full model, run the notebook with num_epochs=25")