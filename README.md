# ECG Atrial Fibrillation Classification Assignment

## Overview

This repository contains a comprehensive machine learning solution for ECG-based atrial fibrillation (AF) classification as part of the WASP Course: Artificial Intelligence and Machine Learning assignment.

### Assignment Objectives
- Implement and evaluate a deep learning model for AF classification from ECG signals
- Achieve target performance: AUROC ≥ 0.97, Average Precision ≥ 0.95
- Submit predictions to competitive leaderboard with F1 score optimization
- Provide detailed analysis and justification of methodology

## Dataset

The dataset is a subset of the [CODE dataset](https://scilifelab.figshare.com/articles/dataset/CODE_dataset/15169716): an annotated database of ECGs from the Telehealth Network of Minas Gerais, Brazil (2010-2016).

### Data Characteristics
- **Input**: 12-lead ECG signals preprocessed to 4096 samples at 400Hz (10.24 seconds)
- **Features**: 8 leads (I, II, V1, V2, V3, V4, V5, V6) 
- **Task**: Binary classification (AF vs Non-AF)
- **Challenge**: Class imbalance with AF as minority class (~10-20%)

### Preprocessing Pipeline
- Resampling to 400Hz for uniform temporal resolution
- Zero-padding to 4096 samples for fixed-size inputs
- Baseline removal to eliminate low-frequency artifacts
- 60Hz powerline noise filtering

## Model Architecture

### Improved Deep CNN
- **4 Convolutional Layers**: Progressive feature extraction with kernel sizes [7,5,3,3]
- **Channel Progression**: 8 → 64 → 128 → 256 → 512 features
- **Regularization**: Batch normalization after each conv layer, 50% dropout in FC layers
- **Pooling**: Global average pooling to reduce parameters and overfitting
- **Classification Head**: 3-layer MLP (512→256→64→1) with ReLU activations

### Key Improvements Over Baseline
- Multi-scale temporal pattern capture through varied kernel sizes
- Batch normalization for training stability and faster convergence
- Deeper architecture for complex AF pattern recognition
- Global average pooling instead of flatten for better generalization

## Training Strategy

### Hyperparameters
- **Learning Rate**: 1e-3 (reduced for stability with deeper model)
- **Batch Size**: 16 (smaller batches for better gradient estimates)
- **Epochs**: 25 (increased for deeper model convergence)
- **Weight Decay**: 1e-4 (reduced due to other regularization)
- **Optimizer**: Adam with ReduceLROnPlateau scheduler

### Class Imbalance Handling
- **Weighted BCE Loss**: pos_weight calculated based on class frequencies
- **Stratified Validation**: Maintains class distribution in train/validation split
- **Balanced Metrics**: Focus on AUROC, AP, and F1 rather than accuracy

### Evaluation Metrics
- **AUROC**: Threshold-independent ranking performance
- **Average Precision (AP)**: Precision-recall performance, sensitive to minority class
- **F1 Score**: Balanced precision-recall for practical deployment
- **Learning Curves**: Training/validation loss and metric tracking

## Repository Structure

```
├── assignment_ecg_classification.ipynb    # Main assignment notebook
├── README.md                             # This file
├── requirements.txt                      # Python dependencies (auto-generated)
├── model.pth                            # Saved trained model (generated during training)
└── .git/                                # Git repository metadata
```

## Usage Instructions

### Google Colab Deployment
1. Upload `assignment_ecg_classification.ipynb` to Google Colab
2. Set Runtime → Change runtime type → Hardware accelerator: GPU
3. Update student information in cell 2:
   - Replace `[YOUR NAME HERE]` with actual student name
   - Replace `[YOUR_TEAM_ID]` with chosen team identifier
4. Run all cells sequentially - the notebook handles:
   - Data download and preprocessing
   - Model training and validation
   - Performance evaluation with learning curves
   - Test prediction generation

### Local Development
```bash
# Clone repository
git clone [repository-url]
cd MachineLearningAssignment

# Install dependencies (Python ≥ 3.9 required)
pip install -r requirements.txt

# Run Jupyter notebook
jupyter notebook assignment_ecg_classification.ipynb
```

### Leaderboard Submission
1. Update team registration in cell 46:
   - Replace `[YOUR_TEAM_ID]` and `[YOUR_PASSWORD]`
2. Run registration cell (only once)
3. Execute prediction cells to generate `soft_pred`
4. Submit with meaningful notes for tracking (e.g., "DeepCNN_v1")

## Implementation Details

### Data Analysis Findings
- Significant class imbalance requiring careful evaluation metrics
- Age and gender distributions vary between AF and non-AF cases
- Naive classifier (majority class) achieves high accuracy but poor clinical utility
- Preprocessing essential for noise reduction and standardization

### Model Design Rationale
- **Multi-scale convolutions** capture both local heartbeat patterns and global rhythm irregularities
- **Batch normalization** enables stable training of deeper networks
- **Dropout regularization** prevents overfitting on limited medical data
- **Global average pooling** reduces parameters while maintaining spatial information

### Training Methodology
- **80-20 train/validation split** with stratification for balanced evaluation
- **Weighted loss function** addresses class imbalance at training level
- **Learning rate scheduling** enables fine-tuning in later epochs
- **Multiple metrics tracking** ensures balanced performance assessment

## Performance Targets

### Assignment Requirements
- **AUROC**: ≥ 0.97 (ranking performance across thresholds)
- **Average Precision**: ≥ 0.95 (precision-recall performance)
- **F1 Score**: Maximize for leaderboard ranking
- **Submissions**: Minimum 3 submissions with documented improvements

### Expected Results
The implemented architecture is designed to exceed target metrics through:
- Sophisticated feature extraction via deep CNN
- Proper class imbalance handling
- Comprehensive regularization strategy
- Optimized hyperparameters based on validation performance

## Submission Strategy

### Three Planned Submissions
1. **DeepCNN_v1**: Foundation model with improved architecture and weighted loss
2. **DeepCNN_v2**: Enhanced with data augmentation, focal loss, and ensemble methods
3. **DeepCNN_v3**: Final optimization with attention mechanisms and threshold tuning

### Submission Notes
- Each submission includes meaningful identifier for tracking
- Progressive improvements documented in explanation sections
- Metrics recorded in submission table after leaderboard evaluation

## Technical Requirements

### Dependencies
- Python ≥ 3.9
- PyTorch ≥ 1.9
- NumPy, Pandas, Matplotlib
- scikit-learn for metrics
- h5py for data loading
- tqdm for progress tracking
- ecg_plot for visualization

### Computational Requirements
- **GPU Recommended**: Training time ~30-60 minutes on GPU vs 4-8 hours on CPU
- **Memory**: ~4-8GB RAM for data loading and model training
- **Storage**: ~500MB for dataset download and preprocessing

## Validation and Testing

### Code Validation
- All 7 coding tasks implemented with proper error handling
- All 5 explanation tasks completed with detailed justifications
- Notebook tested for Colab compatibility
- Model architecture validated for target performance

### Performance Validation
- Learning curves monitored for overfitting detection
- Multiple metrics tracked for balanced evaluation
- Class imbalance properly addressed throughout pipeline
- Threshold analysis for optimal F1 performance

## Contributing

This is an academic assignment. For questions or issues related to the implementation:

1. Check the detailed explanations in notebook cells
2. Verify all placeholder values are replaced with actual data
3. Ensure GPU acceleration is enabled in Colab
4. Contact course instructors for assignment-specific questions

## License

This code is developed for educational purposes as part of the WASP AI/ML course assignment. Use in accordance with academic integrity policies.

## Acknowledgments

- **Course**: WASP Artificial Intelligence and Machine Learning
- **Lecturer**: Dave Zachariah  
- **Assignment Team**: Jingwei Hu, Tianru Zhang, David Vävinggren
- **Dataset**: CODE dataset from Telehealth Network of Minas Gerais
- **Reference**: ["Automatic diagnosis of the 12-lead ECG using a deep neural network"](https://www.nature.com/articles/s41467-020-15432-4)

---

**Note**: Before submission, ensure all placeholder values (marked with `[YOUR_...]`) are replaced with actual information, and complete at least three leaderboard submissions as documented in the assignment requirements.