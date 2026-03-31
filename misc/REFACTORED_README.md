# Refactored Attribution Methods

This folder contains a refactored version of the attribution methods with a clean, modular structure.

## File Structure

### Core Files

1. **`utility.py`** - Common utility functions
   - Data classes: `AttributionResult`, `StepInfo`, `InsDelScores`
   - Model wrapper: `ClassLogitModel`
   - Gradient computation utilities
   - Metric computation functions (Var_nu, CV2, Q)
   - Insertion/Deletion evaluation
   - Optimization functions (μ-optimization, signal-harvesting, path optimization)

2. **`ig.py`** - Standard Integrated Gradients
   - Function: `compute_ig(model, input, params)`
   - Uniform measure μ_k = 1/N, straight-line path

3. **`idig.py`** - Integrated Directed IG (IDGI)
   - Function: `compute_idig(model, input, params)`
   - Weighted measure μ_k ∝ |Δf_k|, straight-line path

4. **`guided_ig.py`** - Guided Integrated Gradients
   - Function: `compute_guided_ig(model, input, params)`
   - Uniform measure, adaptive low-gradient-first path

5. **`lig.py`** - Joint* (Linearized IG with signal harvesting)
   - Function: `compute_lig(model, input, params)`
   - Full signal-harvesting solution with joint γ and μ optimization
   - This is the most advanced method

6. **`lig_idig.py`** - μ-Optimizer (u-Optimizer)
   - Function: `compute_lig_idig(model, input, params)`
   - Optimized measure μ* with signal harvesting, straight-line path

7. **`compare_methods.py`** - Evaluation framework
   - Compare methods across different models
   - Compute insertion/deletion metrics
   - Support for ResNet50, VGG16, DenseNet121, ViT-B-16

## Usage

### Basic Usage - Individual Methods

```python
import torch
from utility import ClassLogitModel, get_device
from ig import compute_ig
from idig import compute_idig
from guided_ig import compute_guided_ig
from lig import compute_lig
from lig_idig import compute_lig_idig

# Setup
device = get_device()
model = ...  # Your pretrained model
target_class = 243  # e.g., "bull mastiff"

# Wrap model to output target class logit
wrapped_model = ClassLogitModel(model, target_class)
wrapped_model.eval()

# Load your image (1, 3, H, W)
x = ...  # Your input image tensor
baseline = torch.zeros_like(x)  # or other baseline

# Define parameters
params = {
    'baseline': baseline,
    'N': 50,  # number of steps
}

# Run IG
result_ig = compute_ig(wrapped_model, x, params)

# Run IDGI
result_idig = compute_idig(wrapped_model, x, params)

# Run Guided IG
result_guided = compute_guided_ig(wrapped_model, x, params)

# Run LIG-IDIG (μ-Optimizer)
params_lig_idig = {
    'baseline': baseline,
    'N': 50,
    'lam': 1.0,
    'tau': 0.01,
    'n_iter': 300,
}
result_lig_idig = compute_lig_idig(wrapped_model, x, params_lig_idig)

# Run LIG (Joint*)
params_lig = {
    'baseline': baseline,
    'N': 50,
    'lam': 1.0,
    'tau': 0.01,
    'G': 16,
    'patch_size': 14,
    'n_alternating': 2,
    'mu_iter': 300,
    'path_iter': 10,
}
result_lig = compute_lig(wrapped_model, x, params_lig)

# Access results
print(f"IG - Q: {result_ig.Q:.4f}, CV²: {result_ig.CV2:.4f}")
print(f"IDGI - Q: {result_idig.Q:.4f}, CV²: {result_idig.CV2:.4f}")
print(f"Guided IG - Q: {result_guided.Q:.4f}, CV²: {result_guided.CV2:.4f}")
print(f"LIG-IDIG - Q: {result_lig_idig.Q:.4f}, CV²: {result_lig_idig.CV2:.4f}")
print(f"LIG - Q: {result_lig.Q:.4f}, CV²: {result_lig.CV2:.4f}")

# Get attributions
attributions = result_lig.attributions  # (1, C, H, W)
```

### Using the Comparison Framework

```bash
# Compare methods on an image with ResNet50
python compare_methods.py --model resnet50 --image path/to/image.jpg

# Compare specific methods
python compare_methods.py --model vgg16 --image path/to/image.jpg \
    --methods ig idig guided_ig lig

# Use a specific model and target class
python compare_methods.py --model densenet121 --image path/to/image.jpg \
    --target-class 281

# Adjust number of steps
python compare_methods.py --model resnet50 --image path/to/image.jpg \
    --steps 100

# Use a specific device
python compare_methods.py --model resnet50 --image path/to/image.jpg \
    --device cuda
```

### Using the Comparison Framework in Code

```python
from compare_methods import compare_methods

results = compare_methods(
    model_name='resnet50',
    image_path='path/to/image.jpg',
    target_class=None,  # Auto-detect
    methods=['ig', 'idig', 'guided_ig', 'lig'],
    metrics=['insertion', 'deletion', 'ins-del'],
    N=50,
    device='cuda',
)

# Results is a list of AttributionResult objects
for result in results:
    print(f"{result.name}: Q={result.Q:.4f}, "
          f"Ins AUC={result.insdel.insertion_auc:.4f}")
```

## Method Parameters

### Standard IG (`ig.py`)
- `baseline`: Baseline tensor (required)
- `N`: Number of steps (default: 50)

### IDGI (`idig.py`)
- `baseline`: Baseline tensor (required)
- `N`: Number of steps (default: 50)

### Guided IG (`guided_ig.py`)
- `baseline`: Baseline tensor (required)
- `N`: Number of steps (default: 50)

### LIG-IDIG / μ-Optimizer (`lig_idig.py`)
- `baseline`: Baseline tensor (required)
- `N`: Number of steps (default: 50)
- `lam`: Signal-harvesting strength λ (default: 1.0)
- `tau`: L2 admissibility (default: 0.01)
- `n_iter`: Optimization iterations (default: 300)

### LIG / Joint* (`lig.py`)
- `baseline`: Baseline tensor (required)
- `N`: Number of steps (default: 50)
- `lam`: Signal-harvesting strength λ (default: 1.0)
- `tau`: L2 admissibility (default: 0.01)
- `G`: Number of spatial groups (default: 16)
- `patch_size`: Patch size for grouping (default: 14)
- `n_alternating`: Alternating iterations (default: 2)
- `mu_iter`: μ optimization iterations (default: 300)
- `path_iter`: Path optimization iterations (default: 10)
- `init_path`: Optional initial path (list of N+1 tensors)

## Model Options in `compare_methods.py`

- **ResNet50**: `--model resnet50`
- **VGG16**: `--model vgg16`
- **DenseNet121**: `--model densenet121`
- **ViT-B-16**: `--model vit_b_16`

## Evaluation Metrics

The framework computes the following metrics:

1. **Q (Quality Score)**: 1 / (1 + CV²), where 1.0 is perfect
2. **CV²**: Coefficient of variation squared of step fidelity
3. **Var_nu**: Variance of step fidelity
4. **Insertion AUC**: Area under insertion curve (higher is better)
5. **Deletion AUC**: Area under deletion curve (lower is better)
6. **Ins-Del**: Insertion AUC - Deletion AUC (higher is better)

## Key Features

1. **Clean API**: Each method has a simple `compute_*(model, input, params)` interface
2. **Self-contained**: No dependencies on original codebase files
3. **Modular**: Easy to add new methods or modify existing ones
4. **Comprehensive**: Includes all utilities for metrics and evaluation
5. **Flexible**: Support for multiple models and evaluation metrics

## Notes

- All methods expect a wrapped model (`ClassLogitModel`) that outputs scalar logits
- Input tensors should be in shape (1, C, H, W)
- Baseline tensors should have the same shape as input
- The default methods list is: `['ig', 'idig', 'guided_ig', 'lig']`
- The default metrics list is: `['insertion', 'deletion', 'ins-del']`
