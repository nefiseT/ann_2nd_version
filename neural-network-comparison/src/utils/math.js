/**
 * Core matrix and vector operations for neural networks
 * Designed for clarity over performance
 */

const Math_ = {
    /**
     * Matrix multiplication: (m x n) × (n x p) → (m x p)
     */
    matmul(a, b) {
        const m = a.length;
        const n = a[0].length;
        const p = b[0].length;
        
        const result = Array(m).fill(0).map(() => Array(p).fill(0));
        
        for (let i = 0; i < m; i++) {
            for (let j = 0; j < p; j++) {
                for (let k = 0; k < n; k++) {
                    result[i][j] += a[i][k] * b[k][j];
                }
            }
        }
        return result;
    },

    /**
     * Add bias vector to each row of matrix
     */
    addBias(matrix, bias) {
        return matrix.map((row, i) => 
            row.map((val, j) => val + bias[i])
        );
    },

    /**
     * Element-wise multiplication (Hadamard product)
     */
    multiply(a, b) {
        return a.map((row, i) =>
            row.map((val, j) => val * b[i][j])
        );
    },

    /**
     * Element-wise addition
     */
    add(a, b) {
        return a.map((row, i) =>
            row.map((val, j) => val + b[i][j])
        );
    },

    /**
     * Scalar multiplication
     */
    scale(matrix, scalar) {
        return matrix.map(row => row.map(val => val * scalar));
    },

    /**
     * Matrix transpose
     */
    transpose(matrix) {
        return matrix[0].map((_, j) =>
            matrix.map(row => row[j])
        );
    },

    /**
     * ReLU activation: max(0, x)
     */
    relu(matrix) {
        return matrix.map(row =>
            row.map(val => Math.max(0, val))
        );
    },

    /**
     * ReLU derivative: 1 if x > 0, else 0
     */
    reluDerivative(matrix) {
        return matrix.map(row =>
            row.map(val => val > 0 ? 1 : 0)
        );
    },

    /**
     * Sigmoid activation: 1 / (1 + e^-x)
     */
    sigmoid(matrix) {
        return matrix.map(row =>
            row.map(val => 1 / (1 + Math.exp(-val)))
        );
    },

    /**
     * Sigmoid derivative: sigmoid(x) * (1 - sigmoid(x))
     */
    sigmoidDerivative(sigmoid) {
        return sigmoid.map(row =>
            row.map(val => val * (1 - val))
        );
    },

    /**
     * Mean Squared Error loss
     */
    mse(predicted, actual) {
        let sum = 0;
        for (let i = 0; i < predicted.length; i++) {
            for (let j = 0; j < predicted[i].length; j++) {
                const diff = predicted[i][j] - actual[i][j];
                sum += diff * diff;
            }
        }
        return sum / (predicted.length * predicted[0].length);
    },

    /**
     * Initialize weights with Xavier initialization
     * Helps with training stability
     */
    initWeights(rows, cols) {
        const limit = Math.sqrt(6 / (rows + cols));
        return Array(rows).fill(0).map(() =>
            Array(cols).fill(0).map(() =>
                Math.random() * 2 * limit - limit
            )
        );
    },

    /**
     * Initialize biases to zero
     */
    initBias(size) {
        return Array(size).fill(0).map(() => [0]);
    }
};
