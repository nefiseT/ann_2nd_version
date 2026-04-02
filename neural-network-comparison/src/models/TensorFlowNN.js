/**
 * Configurable Neural Network using TensorFlow.js
 * Supports flexible hidden layer depth, activation functions, and optimizers
 */

class TensorFlowNN {
    constructor({ inputSize, hiddenLayers = [8], activation = 'relu', learningRate = 0.01, optimizer = 'adam' }) {
        this.trainingHistory = [];
        this.activation = activation;
        this.hiddenLayers = hiddenLayers;
        this.learningRate = learningRate;
        this.optimizerName = optimizer;

        const layers = [];

        // First hidden layer (needs inputShape)
        layers.push(tf.layers.dense({
            inputShape: [inputSize],
            units: hiddenLayers[0],
            activation: activation,
            kernelInitializer: 'glorotUniform'
        }));

        // Additional hidden layers
        for (let i = 1; i < hiddenLayers.length; i++) {
            layers.push(tf.layers.dense({
                units: hiddenLayers[i],
                activation: activation,
                kernelInitializer: 'glorotUniform'
            }));
        }

        // Output layer: always sigmoid for binary classification
        layers.push(tf.layers.dense({
            units: 1,
            activation: 'sigmoid',
            kernelInitializer: 'glorotUniform'
        }));

        this.model = tf.sequential({ layers });

        let opt;
        switch (optimizer) {
            case 'adam':    opt = tf.train.adam(learningRate);    break;
            case 'rmsprop': opt = tf.train.rmsprop(learningRate); break;
            case 'adagrad': opt = tf.train.adagrad(learningRate); break;
            case 'sgd':
            default:        opt = tf.train.sgd(learningRate);     break;
        }

        this.model.compile({
            optimizer: opt,
            loss: 'binaryCrossentropy',
            metrics: []
        });
    }

    /**
     * Train the model
     */
    async train(X, Y, epochs = 100, batchSize = 64) {
        const startTime = performance.now();

        const xs = tf.tensor2d(X);
        const ys = tf.tensor2d(Y);

        // No epoch callback — avoids GPU→CPU sync every epoch, which is a
        // significant bottleneck in TF.js. Loss history is read from the
        // return value instead.
        const result = await this.model.fit(xs, ys, {
            epochs,
            batchSize,
            verbose: 0,
            shuffle: true
        });

        const endTime = performance.now();
        this.trainingTime = endTime - startTime;
        this.trainingHistory = result.history.loss;

        xs.dispose();
        ys.dispose();

        return {
            trainingTime: this.trainingTime,
            finalLoss: this.trainingHistory[this.trainingHistory.length - 1],
            history: this.trainingHistory
        };
    }

    /**
     * Make predictions on input array X
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

        const grid = [];
        for (let i = 0; i < resolution; i++) {
            grid.push(result.slice(i * resolution, (i + 1) * resolution).map(p => p[0]));
        }

        return grid;
    }

    dispose() {
        this.model.dispose();
    }
}
