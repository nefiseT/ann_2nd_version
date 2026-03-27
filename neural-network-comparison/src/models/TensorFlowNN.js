/**
 * Neural Network using TensorFlow.js
 * Same architecture as ManualNN for fair comparison
 * 
 * Demonstrates: using production-ready ML libraries
 */

class TensorFlowNN {
    constructor(inputSize, hiddenSize, outputSize, learningRate = 0.1) {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.outputSize = outputSize;
        this.learningRate = learningRate;
        this.trainingHistory = [];

        // Build sequential model
        this.model = tf.sequential({
            layers: [
                // Input layer (implicit)
                // Hidden layer: ReLU activation
                tf.layers.dense({
                    inputShape: [inputSize],
                    units: hiddenSize,
                    activation: 'relu',
                    kernelInitializer: 'glorotUniform'
                }),
                // Output layer: Sigmoid activation
                tf.layers.dense({
                    units: outputSize,
                    activation: 'sigmoid',
                    kernelInitializer: 'glorotUniform'
                })
            ]
        });

        // Compile with MSE loss and SGD optimizer
        this.model.compile({
            optimizer: tf.train.sgd(learningRate),
            loss: 'meanSquaredError',
            metrics: []
        });
    }

    /**
     * Train the model
     */
    async train(X, Y, epochs = 100, batchSize = 32) {
        const startTime = performance.now();

        // Convert to tensors
        const xs = tf.tensor2d(X);
        const ys = tf.tensor2d(Y);

        // Train with custom callback to track loss
        const history = await this.model.fit(xs, ys, {
            epochs: epochs,
            batchSize: batchSize,
            verbose: 0,
            shuffle: true,
            callbacks: {
                onEpochEnd: (epoch, logs) => {
                    this.trainingHistory.push(logs.loss);
                    if ((epoch + 1) % 10 === 0) {
                        console.log(`[TensorFlowNN] Epoch ${epoch + 1}/${epochs}, Loss: ${logs.loss.toFixed(6)}`);
                    }
                }
            }
        });

        const endTime = performance.now();
        this.trainingTime = endTime - startTime;

        // Clean up tensors
        xs.dispose();
        ys.dispose();

        return {
            trainingTime: this.trainingTime,
            finalLoss: this.trainingHistory[this.trainingHistory.length - 1],
            history: this.trainingHistory
        };
    }

    /**
     * Make predictions
     */
    predict(X) {
        const xs = tf.tensor2d(X);
        const predictions = this.model.predict(xs);
        const result = predictions.arraySync();
        xs.dispose();
        predictions.dispose();
        return result;
    }

    /**
     * Get predictions on a grid for visualization
     */
    predictGrid(resolution = 50) {
        const gridPoints = [];
        const step = 2 / resolution;

        for (let y = -1; y < 1; y += step) {
            for (let x = -1; x < 1; x += step) {
                gridPoints.push([x, y]);
            }
        }

        const xs = tf.tensor2d(gridPoints);
        const predictions = this.model.predict(xs);
        const result = predictions.arraySync();
        xs.dispose();
        predictions.dispose();

        // Reshape to 2D grid
        const grid = [];
        for (let i = 0; i < resolution; i++) {
            grid.push(result.slice(i * resolution, (i + 1) * resolution).map(p => p[0]));
        }

        return grid;
    }

    /**
     * Cleanup: dispose of model and tensors
     */
    dispose() {
        this.model.dispose();
    }
}
