# Neural Network Comparison

Compare two independently configured TensorFlow.js neural networks side by side on a circle classification task. Both models train on the same dataset so results reflect only the difference in architecture and hyperparameters.

## Controls

### Shared Environment
Both models receive the same data, generated once before training starts.

| Control | Range | Description |
|---------|-------|-------------|
| Dataset Size | 100 – 2000 | Number of samples generated |
| Noise | 0.00 – 0.50 | Jitter applied to coordinates before labeling — creates an ambiguous boundary |
| Epochs | 10 – 500 | Training epochs (same for both models) |

### Per-Model (Model A & Model B)

| Control | Options | Description |
|---------|---------|-------------|
| Learning Rate | 0.0001 – 0.5 | Step size for weight updates |
| Optimizer | Adam, SGD, RMSProp, Adagrad | Optimization algorithm |
| Activation | ReLU, Sigmoid, Tanh, ELU, SELU | Hidden layer activation function |
| Hidden Layers | Comma-separated integers | One number per layer, e.g. `16, 8` builds two layers |

### Results
After training each model card shows **Final Loss**, **Accuracy**, **Training Time**, and **Epochs Trained**. The comparison panel below displays a config diff and a color-coded winner table.

## Task

Points are sampled uniformly from [−1, 1] × [−1, 1]. A point is labeled **1** (inside) if it falls within a circle of radius 0.6 centered at the origin, **0** (outside) otherwise. With noise > 0, coordinate jitter is applied before the label is assigned, making the boundary fuzzy.

## Structure

```
neural-network-comparison/
├── index.html              # Browser entry point
├── src/
│   ├── models/
│   │   └── TensorFlowNN.js # Configurable TF.js model (both A and B)
│   ├── utils/
│   │   ├── dataset.js      # Circle dataset generation with noise
│   │   ├── metrics.js      # Accuracy, MSE, precision, recall
│   │   └── math.js         # Matrix operations (reference)
│   ├── ui/
│   │   ├── canvas.js       # Heatmap + data point rendering
│   │   └── controls.js     # Hyperparameter UI handlers
│   └── main.js             # Training orchestration & comparison
```

## Run

Open `index.html` in a browser — no build step or server required.
