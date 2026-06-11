# Hybrid Quantum-Classical Image Classifier

> A **production-ready** hybrid quantum-classical machine learning system combining classical deep learning and quantum computing for image classification on MNIST digits.

**Status**: ✅ Complete and Ready to Deploy  
**Last Updated**: June 11, 2026

---

## 🎯 Project Overview

This project demonstrates a cutting-edge approach to image classification by combining:

- **Classical Component**: PyTorch CNN that extracts interpretable features from images (28×28 → 4×4 = 16 features)
- **Quantum Component**: PennyLane-based Variational Quantum Classifier (VQC) that acts as the final decision-maker
- **Benchmarking**: Complete comparison between classical baseline and hybrid model

Perfect for researchers, students, and practitioners learning quantum machine learning concepts.

### Key Achievements

✅ Full end-to-end training pipeline  
✅ Reproducible results with comprehensive documentation  
✅ Educational Jupyter notebook with visualizations  
✅ Production-grade code with proper error handling  
✅ Comparison metrics and performance analysis  
✅ Ready-to-deploy model artifacts  

---

## 🚀 Quick Start (5 Minutes)

### Step 1: Clone & Install

```bash
# Clone repository
git clone https://github.com/shauravkhadka/Hybrid-Quantum-Classical-Image-Classifier-.git
cd Hybrid-Quantum-Classical-Image-Classifier-

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Run Jupyter Notebook

```bash
jupyter notebook notebooks/Data_Preprocessing_Pipeline.ipynb
```

This interactive notebook shows:
- Data loading and exploration
- Feature extraction pipeline
- Visualization of learned features
- Comparison of extraction methods

### Step 3: Train Models

```bash
# Train classical baseline (30-60 seconds)
python scripts/train_classical.py --num_epochs 20

# Train hybrid model (5-10 minutes)
python scripts/train_hybrid.py --num_epochs 20 --n_qubits 4 --n_layers 2
```

### Step 4: Compare Results

```bash
python scripts/evaluate_comparison.py
```

Generates:
- Performance metrics table
- Training curves
- Comparison visualizations → `results/comparison.png`

---

## 📊 Architecture

### Pipeline Overview

```
Input Image (28×28 pixels)
    ↓
[Classical Feature Extractor]
    • Conv2d layer 1: 32 filters
    • Conv2d layer 2: 64 filters
    • Adaptive pooling → 4×4 feature map
    • Dense layers: (1024 → 128 → 16)
    ↓
Normalized Features (16 dimensions ∈ [0,1])
    ↓
[Quantum Circuit (4 qubits)]
    • Angle encoding: RY(π × feature_i)
    • Variational layer 1: RZ rotations + CNOT ladder
    • Variational layer 2: RZ rotations + CNOT ladder
    • Measurement: Z expectation on qubit 0
    ↓
Binary Classification (P(class 1) ∈ [0,1])
```

### Classical Feature Extractor

**Input**: 28×28 grayscale images  
**Process**:
- Conv2d(1→32, kernel=3) + ReLU + MaxPool
- Conv2d(32→64, kernel=3) + ReLU + MaxPool
- AdaptiveAvgPool → 4×4
- Flatten + Dense(1024→128) + Dense(128→16)
- Sigmoid normalization to [0,1]

**Output**: 16 normalized feature values

### Quantum Classifier

**Qubits**: 4 (one per feature group)  
**Depth**: 2 variational layers  
**Trainable Parameters**: 16 (8 per layer)

```
Encoding: |ψ_enc⟩ = ∏ᵢ RY(π·xᵢ) |0⟩
Variational: |ψ_var⟩ = U₂(θ) CNOT U₁(θ) |ψ_enc⟩
Measurement: P(1) = ⟨Z₀⟩ → (result+1)/2
```

---

## 📁 Repository Structure

```
├── README.md                                    # This file
├── requirements.txt                             # Python dependencies
├── notebooks/
│   └── Data_Preprocessing_Pipeline.ipynb       # Educational tutorial
├── src/
│   ├── __init__.py                             # Package init
│   ├── feature_extractor.py                    # Classical CNN & PCA
│   │   ├── ClassicalCNNFeatureExtractor       # Main feature extractor
│   │   ├── PCAFeatureExtractor                # Alternative method
│   │   └── HybridFeatureExtractor             # Wrapper
│   ├── quantum_classifier.py                   # Quantum circuits
│   │   ├── QuantumCircuit                     # Base PQC
│   │   ├── VariationalQuantumClassifier       # PyTorch wrapper
│   │   └── HybridQuantumClassifier            # End-to-end model
│   ├── hybrid_model.py                         # Integration layer
│   │   ├── HybridModel                        # Complete pipeline
│   │   ├── TrainingConfig                     # Configuration
│   │   └── create_hybrid_model()              # Factory function
│   └── utils.py                                # Training utilities
│       ├── load_mnist_data()
│       ├── train_epoch()
│       ├── evaluate()
│       ├── compute_metrics()
│       └── plot_*() functions
├── scripts/
│   ├── train_classical.py                      # Classical baseline training
│   ├── train_hybrid.py                         # Hybrid model training
│   └── evaluate_comparison.py                  # Performance comparison
├── data/                                        # MNIST (auto-downloaded)
├── models/                                      # Model checkpoints
│   ├── classical_best.pth
│   └── hybrid_best.pth
└── results/                                     # Training results
    ├── classical_results.json
    ├── hybrid_results.json
    └── comparison.png
```

---

## 📈 Performance Benchmark

### Expected Results

| Metric | Classical CNN | Hybrid Model |
|--------|--------------|--------------|
| **Accuracy** | ~98% | ~94-96% |
| **Precision** | 0.98 | 0.94-0.96 |
| **Recall** | 0.98 | 0.94-0.96 |
| **F1-Score** | 0.98 | 0.94-0.96 |
| **Training Time** | ~45s (20 epochs) | ~5-10min (20 epochs) |
| **Inference Time** | ~2ms/batch | ~15-20ms/batch |

### Why Hybrid is Slower?

The trade-off is intentional and educational:
1. Classical CNN works perfectly for this task (it's designed for it)
2. Quantum circuit execution (even simulated) is expensive
3. On real quantum hardware, quantum advantage could emerge for certain problem types
4. This project demonstrates the *integration* rather than quantum speedup

---

## 💻 Detailed Usage Guide

### 1. Understanding the Feature Extraction

Open the Jupyter notebook to visualize:

```bash
jupyter notebook notebooks/Data_Preprocessing_Pipeline.ipynb
```

**Sections**:
1. Load MNIST data
2. Explore sample images
3. Train CNN feature extractor
4. Extract and visualize features
5. Compare with PCA method
6. Demonstrate quantum encoding

### 2. Training Classical Baseline

```bash
python scripts/train_classical.py \
  --learning_rate 1e-3 \
  --num_epochs 20 \
  --batch_size 32 \
  --patience 5 \
  --use_cuda
```

**Output**:
- `models/classical_best.pth` - Best model checkpoint
- `results/classical_results.json` - Metrics and history

### 3. Training Hybrid Model

```bash
python scripts/train_hybrid.py \
  --learning_rate 1e-3 \
  --num_epochs 20 \
  --batch_size 32 \
  --n_qubits 4 \
  --n_layers 2 \
  --feature_method cnn \
  --n_features 16 \
  --use_cuda
```

**Options**:
- `--feature_method`: `cnn` (default) or `pca`
- `--n_qubits`: 2-6 qubits (4 is balanced)
- `--n_layers`: 1-3 variational layers

**Output**:
- `models/hybrid_best.pth` - Best model checkpoint
- `results/hybrid_results.json` - Metrics and history

### 4. Compare Results

```bash
python scripts/evaluate_comparison.py
```

**Output**:
- Console table with side-by-side metrics
- `results/comparison.png` with visualizations

---

## 🔧 Using as a Library

### Import and Create Model

```python
from src import create_hybrid_model

# Create model
model = create_hybrid_model(
    n_qubits=4,
    n_layers=2,
    feature_method='cnn',
    n_features=16,
    use_cuda=False
)

# Print architecture
print(model)
```

### Use for Inference

```python
import torch

# Your image (28×28)
image = torch.randn(1, 1, 28, 28)

# Get prediction
with torch.no_grad():
    prediction = model(image)  # Returns probability

print(f"Class 0 probability: {(1-prediction).item():.2%}")
print(f"Class 1 probability: {prediction.item():.2%}")
```

### Load Pre-trained Model

```python
model = create_hybrid_model()
model.load_state_dict(torch.load('models/hybrid_best.pth'))
model.eval()
```

---

## 📚 Technical Details

### Feature Extraction Methods

#### Method 1: Classical CNN (Recommended)
- **Pros**: Learns task-optimal features, high separability
- **Cons**: Requires training, larger memory footprint
- **Use**: Default for best results

#### Method 2: PCA
- **Pros**: Fast, deterministic, interpretable
- **Cons**: Linear transformations only, may underutilize quantum circuit
- **Use**: Comparison and analysis

### Quantum Circuit Configuration

**4-Qubit Circuit (Default)**:
```
Qubit 0 ──RY(π·x₀)──RZ(θ₀)──●──RZ(θ₈)──●──┤ Z ⟩
Qubit 1 ──RY(π·x₁)──RZ(θ₁)──┼──RZ(θ₉)──┼──┤
Qubit 2 ──RY(π·x₂)──RZ(θ₂)──┼─RZ(θ₁₀)──┼──┤
Qubit 3 ──RY(π·x₃)──RZ(θ₃)──X──RZ(θ₁₁)──X──┤
```

**Trainable Parameters**: 16 (8 per layer)  
**Entanglement**: CNOT ladder with wrap-around  
**Output**: Pauli-Z expectation → probability

---

## 🎓 Educational Content

### What You'll Learn

1. **Classical ML**: CNN architecture, feature extraction, PyTorch training
2. **Quantum ML**: Parameterized circuits, variational algorithms, quantum encoding
3. **Hybrid Systems**: Integration patterns, data flow between classical and quantum
4. **ML Engineering**: Training pipelines, metrics, reproducibility
5. **Benchmarking**: Performance analysis, visualization, comparison

### Project Flow

```
Data Loading
    ↓
Feature Extraction (Classical CNN)
    ↓
Feature Normalization (0-1 range)
    ↓
Quantum Encoding (Angles: 0 to π)
    ↓
Parameterized Quantum Circuit (Training)
    ↓
Measurement & Classification
    ↓
Loss Computation & Backpropagation
    ↓
Performance Analysis & Comparison
```

---

## 🔬 Extending the Project

### 1. Multi-class Classification

Replace binary with 10-class MNIST:

```python
# In train_hybrid.py, line 45
train_loader, test_loader = load_mnist_data(
    binary=False,  # Changed
    num_classes=10  # Added
)

# Use categorical features: one qubit per digit
model = create_hybrid_model(n_qubits=10)
```

### 2. Different Quantum Backends

```python
# Use Qiskit Aer simulator
import pennylane as qml
dev = qml.device('qiskit.aer', device='aer_simulator')

# Or IBM Quantum hardware
dev = qml.device('qiskit.ibm', 'ibm_backend_name')
```

### 3. Hardware-Efficient Circuits

Modify `src/quantum_classifier.py`:
```python
def _circuit(self, inputs, weights):
    # Add barren plateau mitigation
    # Adjust depth/width for hardware constraints
    # Add error mitigation strategies
```

### 4. Transfer Learning

Pre-train CNN on larger dataset:
```python
model.freeze_feature_extractor()  # Keep CNN weights
# Only train quantum parameters on MNIST
```

### 5. Ensemble Methods

Combine multiple quantum classifiers:
```python
ensemble = [create_hybrid_model() for _ in range(5)]
# Train and aggregate predictions
```

---

## 📊 Results Analysis

### Performance Comparison Example

```
==================================================
HYBRID QUANTUM-CLASSICAL CLASSIFIER - PERFORMANCE COMPARISON
==================================================

        Metric  Classical    Hybrid  Difference  % Change
      Accuracy     0.9800    0.9520      -0.0280    -2.86%
      Precision     0.9800    0.9510      -0.0290    -2.96%
         Recall     0.9800    0.9620      -0.0180    -1.84%
       F1-Score     0.9800    0.9565      -0.0235    -2.40%
      Test Loss     0.0654    0.1123      +0.0469    +71.74%

==================================================
ADDITIONAL METRICS
==================================================
Hybrid Inference Time: 16.45 ms/batch

Hybrid Model Configuration:
  n_qubits: 4
  n_layers: 2
  feature_method: cnn
  n_features: 16
==================================================
```

---

## ⚠️ Common Issues & Troubleshooting

### Issue: Out of Memory

**Solution**: Reduce batch size
```bash
python scripts/train_hybrid.py --batch_size 16  # Default: 32
```

### Issue: Quantum Circuit Too Slow

**Solution**: Reduce number of qubits/layers
```bash
python scripts/train_hybrid.py --n_qubits 2 --n_layers 1
```

### Issue: Poor Convergence

**Solution**: Adjust learning rate
```bash
python scripts/train_hybrid.py --learning_rate 5e-4
```

### Issue: CUDA Not Available

**Solution**: Use CPU (slower but works)
```bash
# Remove --use_cuda flag or run on CPU-only machine
python scripts/train_hybrid.py
```

---

## 📦 Dependencies

Core packages (see `requirements.txt`):

| Package | Version | Purpose |
|---------|---------|---------|
| torch | 2.0.1 | Deep learning framework |
| torchvision | 0.15.2 | Computer vision utilities |
| pennylane | 0.31.0 | Quantum machine learning |
| pennylane-qiskit | 0.31.0 | Qiskit integration |
| qiskit | 0.42.1 | Quantum computing toolkit |
| numpy | 1.24.3 | Numerical computing |
| scikit-learn | 1.3.0 | Machine learning utilities |
| matplotlib | 3.7.2 | Plotting |
| jupyter | 1.0.0 | Interactive notebooks |

---

## 🤝 Contributing

Contributions welcome! Please:

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/your-feature`
3. **Commit** changes: `git commit -m 'Add feature'`
4. **Push** to branch: `git push origin feature/your-feature`
5. **Open** a Pull Request

### Contribution Ideas

- [ ] Add more quantum backends (IonQ, Rigetti)
- [ ] Implement variational optimization strategies
- [ ] Add barren plateau mitigation techniques
- [ ] Support for other datasets
- [ ] Visualization tools
- [ ] Performance profiling
- [ ] Docker container

---

## 📝 Citation

If you use this project:

```bibtex
@software{hybrid_quantum_classifier,
  author = {Shaurav Khadka},
  title = {Hybrid Quantum-Classical Image Classifier},
  year = {2026},
  url = {https://github.com/shauravkhadka/Hybrid-Quantum-Classical-Image-Classifier-},
  note = {GitHub repository}
}
```

---

## 📄 License

MIT License - Free for academic and commercial use

```
MIT License

Copyright (c) 2026 Shaurav Khadka

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

---

## 📞 Support & Contact

- **GitHub Issues**: [Report issues](https://github.com/shauravkhadka/Hybrid-Quantum-Classical-Image-Classifier-/issues)
- **Discussions**: [Ask questions](https://github.com/shauravkhadka/Hybrid-Quantum-Classical-Image-Classifier-/discussions)
- **Email**: shauravkhadka@github.com
- **Portfolio**: [GitHub Profile](https://github.com/shauravkhadka)

---

## 🙏 Acknowledgments

Built with:
- **PennyLane** by Xanadu for quantum machine learning
- **Qiskit** by IBM for quantum computing
- **PyTorch** by Meta for deep learning
- **MNIST** dataset creators (Yann LeCun et al.)

---

## 🎯 Roadmap

### v1.0 (Current)
- ✅ Hybrid quantum-classical classifier
- ✅ Classical baseline comparison
- ✅ Educational Jupyter notebook
- ✅ Performance benchmarking

### v1.1 (Planned)
- [ ] Multi-class classification
- [ ] Transfer learning support
- [ ] Ensemble methods
- [ ] Hardware backend support

### v2.0 (Future)
- [ ] Adaptive circuit design
- [ ] Quantum data encoding
- [ ] Real quantum hardware deployment
- [ ] Advanced error mitigation

---

## 🚀 Quick Reference

```bash
# One-liner to get started
git clone https://github.com/shauravkhadka/Hybrid-Quantum-Classical-Image-Classifier-.git && \
cd Hybrid-Quantum-Classical-Image-Classifier- && \
pip install -r requirements.txt && \
python scripts/train_classical.py && \
python scripts/train_hybrid.py && \
python scripts/evaluate_comparison.py
```

---

**Status**: ✅ Production Ready  
**Python**: 3.8+  
**License**: MIT  
**Last Updated**: June 11, 2026  
**Maintained by**: [@shauravkhadka](https://github.com/shauravkhadka)

---

**⭐ If you find this project useful, please star it on GitHub!**
