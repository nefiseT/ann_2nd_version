# Neural Network Comparison

This project compares a from-scratch manual neural network implementation with TensorFlow.js on a circle classification task.

## Structure
- `index.html`: Browser entry point
- `src/models/`: NN implementations
  - `ManualNN.js`: From-scratch implementation
  - `TensorFlowNN.js`: TensorFlow.js wrapper
- `src/utils/`: Utilities
  - `dataset.js`: Circle generation
  - `math.js`: Matrix operations
  - `metrics.js`: Loss, accuracy calculations
- `src/ui/`: Visualization
  - `canvas.js`: Canvas rendering
  - `controls.js`: Hyperparameter UI
- `src/main.js`: Orchestration

## Run
Open `index.html` in a browser.
