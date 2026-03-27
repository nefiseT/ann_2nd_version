/**
 * Feedforward Neural Network implemented from scratch
 * Demonstrates core ML concepts: forward pass, backpropagation, weight updates
 * 
 * Architecture: input → hidden (ReLU) → output (Sigmoid)
 */

class ManualNN {
    constructor(inputSize, hiddenSize, outputSize, learningRate = 0.1) {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.outputSize = outputSize;
        this.learningRate = learningRate;

        // Initialize weights and biases
        // Layer 1: input → hidden
        this.W1 = Math_.initWeights(inputSize, hiddenSize);
        this.b1 = Math_.initBias(hiddenSize);

        // Layer 2: hidden → output
        this.W2 = Math_.initWeights(hiddenSize, outputSize);
        this.b2 = Math_.initBias(outputSize);

        // Store activations for backpropagation
        this.cache = {};
        this.trainingHistory = [];
    }

    /**
     * Forward pass through the network
     * Input: [[x1, y1], [x2, y2], ...] (batch_size × input_size)
     * Output: [[p1], [p2], ...] (batch_size × output_size)
     */
    forward(X) {
        // Ensure X is 2D array
        if (!Array.isArray(X[0])) {
            X = X.map(x => [x]);
        }

        // Layer 1: Linear transformation + bias
        const Z1 = Math_.matmul(X, this.W1);
        const Z1_biased = Z1.map((row, i) =>
            row.map((val, j) => val + this.b1[j][0])
        );

        // Layer 1: ReLU activation
        const A1 = Math_.relu(Z1_biased);

        // Layer 2: Linear transformation + bias
        const Z2 = Math_.matmul(A1, this.W2);
        const Z2_biased = Z2.map((row, i) =>
            row.map((val, j) => val + this.b2[j][0])
        );

        // Layer 2: Sigmoid activation (output)
        const A2 = Math_.sigmoid(Z2_biased);

        // Cache for backpropagation
        this.cache = { X, Z1_biased, A1, Z2_biased, A2 };

        return A2;
    }

    /**
     * Backward pass: compute gradients using backpropagation
     */
    backward(Y) {
        const { X, Z1_biased, A1, Z2_biased, A2 } = this.cache;
        const m = X.length; // batch size

        // Output layer error: dL/dA2 = (A2 - Y) for MSE loss
        const dA2 = A2.map((row, i) =>
            row.map((val, j) => val - Y[i][j])
        );

        // Sigmoid derivative
        const sigmoidDeriv = Math_.sigmoidDerivative(A2);
        const dZ2 = Math_.multiply(dA2, sigmoidDeriv);

        // Gradients for W2 and b2
        const dW2 = Math_.matmul(Math_.transpose(A1), dZ2);
        const dW2_avg = Math_.scale(dW2, 1 / m);

        const db2 = dZ2.map(row => [row.reduce((a, b) => a + b, 0) / m]);

        // Backprop to hidden layer
        const dA1 = Math_.matmul(dZ2, Math_.transpose(this.W2));

        // ReLU derivative
        const reluDeriv = Math_.reluDerivative(Z1_biased);
        const dZ1 = Math_.multiply(dA1, reluDeriv);

        // Gradients for W1 and b1
        const dW1 = Math_.matmul(Math_.transpose(X), dZ1);
        const dW1_avg = Math_.scale(dW1, 1 / m);

        const db1 = dZ1.map(row => [row.reduce((a, b) => a + b, 0) / m]);

        return { dW1: dW1_avg, db1, dW2: dW2_avg, db2 };
    }

    /**
     * Update weights and biases using gradient descent
     */
    updateWeights(gradients) {
        const { dW1, db1, dW2, db2 } = gradients;

        // Update layer 1
        this.W1 = Math_.add(
            this.W1,
            Math_.scale(dW1, -this.learningRate)
        );
        this.b1 = Math_.add(
            this.b1,
            Math_.scale(db1, -this.learningRate)
        );

        // Update layer 2
        this.W2 = Math_.add(
            this.W2,
            Math_.scale(dW2, -this.learningRate)
        );
        this.b2 = Math_.add(
            this.b2,
            Math_.scale(db2, -this.learningRate)
        );
    }

    /**
     * Train the network on a dataset
     */
    train(X, Y, epochs = 100, batchSize = 32) {
        const startTime = performance.now();
        const n = X.length;

        for (let epoch = 0; epoch < epochs; epoch++) {
            let epochLoss = 0;
            let batchCount = 0;

            // Shuffle data
            const indices = Array.from({ length: n }, (_, i) => i);
            for (let i = n - 1; i > 0; i--) {
                const j = Math.floor(Math.random() * (i + 1));
                [indices[i], indices[j]] = [indices[j], indices[i]];
            }

            // Mini-batch training
            for (let i = 0; i < n; i += batchSize) {
                const batchIndices = indices.slice(i, Math.min(i + batchSize, n));
                const batchX = batchIndices.map(idx => X[idx]);
                const batchY = batchIndices.map(idx => Y[idx]);

                // Forward pass
                const predictions = this.forward(batchX);

                // Compute loss
                const loss = Math_.mse(predictions, batchY);
                epochLoss += loss;
                batchCount++;

                // Backward pass
                const gradients = this.backward(batchY);

                // Update weights
                this.updateWeights(gradients);
            }

            const avgLoss = epochLoss / batchCount;
            this.trainingHistory.push(avgLoss);

            // Log progress every 10 epochs
            if ((epoch + 1) % 10 === 0) {
                console.log(`[ManualNN] Epoch ${epoch + 1}/${epochs}, Loss: ${avgLoss.toFixed(6)}`);
            }
        }

        const endTime = performance.now();
        this.trainingTime = endTime - startTime;

        return {
            trainingTime: this.trainingTime,
            finalLoss: this.trainingHistory[this.trainingHistory.length - 1],
            history: this.trainingHistory
        };
    }

    /**
     * Make predictions on new data
     */
    predict(X) {
        if (!Array.isArray(X[0])) {
            X = X.map(x => [x]);
        }
        return this.forward(X);
    }

    /**
     * Get predictions on a grid for visualization
     */
    predictGrid(resolution = 50) {
        const predictions = [];
        const step = 2 / resolution; // Range is [-1, 1]

        for (let y = -1; y < 1; y += step) {
            const row = [];
            for (let x = -1; x < 1; x += step) {
                const pred = this.predict([[x, y]]);
                row.push(pred[0][0]);
            }
            predictions.push(row);
        }

        return predictions;
    }
}
