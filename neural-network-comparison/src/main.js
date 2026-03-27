/**
 * Main application orchestration
 * Coordinates dataset generation, model training, and visualization
 */

const App = {
    manualNN: null,
    tfNN: null,
    dataset: null,
    trainData: null,
    testData: null,

    init() {
        console.log('🧠 Neural Network Comparison Tool');
        Controls.init();
        console.log('✅ Controls initialized');
    },

    /**
     * Main training function - orchestrates both models
     */
    async train(params) {
        const {
            datasetSize,
            learningRate,
            hiddenSize,
            epochs
        } = params;

        console.log('\n📊 Generating dataset...');
        // Generate dataset
        const { data, labels } = Dataset.generateCircle(datasetSize);
        const split = Dataset.split(data, labels, 0.8);
        
        this.trainData = split.train;
        this.testData = split.test;

        console.log(`✅ Dataset created: ${datasetSize} samples (${Math.floor(datasetSize * 0.8)} train, ${Math.floor(datasetSize * 0.2)} test)`);

        // Initialize models
        console.log('\n🔧 Initializing models...');
        this.manualNN = new ManualNN(2, hiddenSize, 1, learningRate);
        this.tfNN = new TensorFlowNN(2, hiddenSize, 1, learningRate);
        console.log('✅ Models initialized');

        // Train Manual NN
        console.log('\n🚂 Training Manual Neural Network...');
        const manualResult = this.manualNN.train(
            this.trainData.data,
            this.trainData.labels,
            epochs,
            32
        );
        console.log(`✅ Manual NN trained in ${manualResult.trainingTime.toFixed(2)}ms`);

        // Train TensorFlow NN
        console.log('\n🚂 Training TensorFlow.js Model...');
        const tfResult = await this.tfNN.train(
            this.trainData.data,
            this.trainData.labels,
            epochs,
            32
        );
        console.log(`✅ TensorFlow.js trained in ${tfResult.trainingTime.toFixed(2)}ms`);

        // Evaluate on test set
        console.log('\n📈 Evaluating on test set...');
        this.evaluateModels();

        // Visualize predictions
        console.log('\n🎨 Visualizing predictions...');
        this.visualize();

        // Compare results
        console.log('\n📊 Comparing results...');
        this.compare();

        console.log('\n✅ Training complete!\n');
    },

    evaluateModels() {
        // Manual NN evaluation
        const manualPreds = this.manualNN.predict(this.testData.data);
        const manualAccuracy = Metrics.accuracy(manualPreds, this.testData.labels);
        this.manualAccuracy = manualAccuracy;

        console.log(`Manual NN - Accuracy: ${(manualAccuracy * 100).toFixed(2)}%`);

        // TensorFlow NN evaluation
        const tfPreds = this.tfNN.predict(this.testData.data);
        const tfAccuracy = Metrics.accuracy(tfPreds, this.testData.labels);
        this.tfAccuracy = tfAccuracy;

        console.log(`TensorFlow.js - Accuracy: ${(tfAccuracy * 100).toFixed(2)}%`);

        // Update UI
        document.getElementById('manualAccuracy').textContent = `${(manualAccuracy * 100).toFixed(2)}%`;
        document.getElementById('tfAccuracy').textContent = `${(tfAccuracy * 100).toFixed(2)}%`;

        // Update training stats
        Controls.updateStats(this.manualNN, 'manual');
        Controls.updateStats(this.tfNN, 'tensorflow');
    },

    visualize() {
        // Get predictions on grid
        const manualGrid = this.manualNN.predictGrid(50);
        const tfGrid = this.tfNN.predictGrid(50);

        // Draw on canvas
        Canvas.draw('canvasManual', manualGrid, this.testData.data, this.testData.labels);
        Canvas.draw('canvasTensorFlow', tfGrid, this.testData.data, this.testData.labels);

        console.log('✅ Visualizations complete');
    },

    compare() {
        const manualLoss = this.manualNN.trainingHistory[this.manualNN.trainingHistory.length - 1];
        const tfLoss = this.tfNN.trainingHistory[this.tfNN.trainingHistory.length - 1];
        const manualTime = this.manualNN.trainingTime;
        const tfTime = this.tfNN.trainingTime;

        const speedup = (manualTime / tfTime).toFixed(2);
        const lossImprovement = ((manualLoss - tfLoss) / manualLoss * 100).toFixed(2);

        let comparisonHTML = `
            <div class="insight">
                <strong>📊 Performance Summary</strong><br>
                Manual NN Loss: ${manualLoss.toFixed(6)} | TensorFlow Loss: ${tfLoss.toFixed(6)}<br>
                Manual NN Accuracy: ${(this.manualAccuracy * 100).toFixed(2)}% | TensorFlow Accuracy: ${(this.tfAccuracy * 100).toFixed(2)}%<br>
                Manual NN Time: ${manualTime.toFixed(2)}ms | TensorFlow Time: ${tfTime.toFixed(2)}ms
            </div>

            <div class="insight">
                <strong>⚡ Efficiency Analysis</strong><br>
                TensorFlow.js is <strong>${speedup}x faster</strong> than the manual implementation.<br>
                ${tfLoss < manualLoss ? 
                    `TensorFlow achieved <strong>${lossImprovement}%</strong> lower loss.` :
                    `Manual NN achieved <strong>${Math.abs(lossImprovement)}%</strong> lower loss.`
                }
            </div>

            <div class="insight">
                <strong>💡 Why the difference?</strong><br>
                <ul style="margin-left: 20px;">
                    <li><strong>Speed:</strong> TensorFlow.js uses optimized WebGL/WASM backends vs pure JS loops</li>
                    <li><strong>Accuracy:</strong> TensorFlow uses Adam optimizer (adaptive learning) vs vanilla SGD</li>
                    <li><strong>Code:</strong> Manual NN demonstrates core concepts; TensorFlow is production-ready</li>
                </ul>
            </div>

            <div class="insight">
                <strong>🎯 Key Takeaways</strong><br>
                <ul style="margin-left: 20px;">
                    <li>Both models learn the circle boundary successfully</li>
                    <li>TensorFlow.js abstracts away low-level optimization details</li>
                    <li>Manual implementation is educational but slower for real applications</li>
                    <li>For production: use libraries; for learning: implement from scratch</li>
                </ul>
            </div>
        `;

        document.getElementById('comparisonText').innerHTML = comparisonHTML;
    }
};

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    App.init();
});
