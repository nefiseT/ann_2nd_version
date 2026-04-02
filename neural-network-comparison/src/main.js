/**
 * Main application — coordinates dataset generation, model training, and visualization
 */

const App = {
    modelA: null,
    modelB: null,
    testData: null,
    accuracyA: null,
    accuracyB: null,

    init() {
        Controls.init();
    },

    async train({ shared, modelA: cfgA, modelB: cfgB }) {
        const { datasetSize, noise, epochs } = shared;

        // Generate shared dataset
        const { data, labels } = Dataset.generateCircle(datasetSize, 0.6, noise);
        const split = Dataset.split(data, labels, 0.8);
        this.testData = split.test;

        // Dispose previous models
        if (this.modelA) this.modelA.dispose();
        if (this.modelB) this.modelB.dispose();

        // Build models
        this.modelA = new TensorFlowNN({ inputSize: 2, ...cfgA });
        this.modelB = new TensorFlowNN({ inputSize: 2, ...cfgB });

        // Train both sequentially (TF.js shares the GPU context)
        await this.modelA.train(split.train.data, split.train.labels, epochs, 64);
        await this.modelB.train(split.train.data, split.train.labels, epochs, 64);

        this.evaluate();
        this.visualize();
        this.compare(cfgA, cfgB);
    },

    evaluate() {
        const predsA = this.modelA.predict(this.testData.data);
        const predsB = this.modelB.predict(this.testData.data);

        this.accuracyA = Metrics.accuracy(predsA, this.testData.labels);
        this.accuracyB = Metrics.accuracy(predsB, this.testData.labels);

        document.getElementById('aAccuracy').textContent = `${(this.accuracyA * 100).toFixed(2)}%`;
        document.getElementById('bAccuracy').textContent = `${(this.accuracyB * 100).toFixed(2)}%`;

        Controls.updateStats(this.modelA, 'a');
        Controls.updateStats(this.modelB, 'b');
    },

    visualize() {
        Canvas.draw('canvasA', this.modelA.predictGrid(50), this.testData.data, this.testData.labels);
        Canvas.draw('canvasB', this.modelB.predictGrid(50), this.testData.data, this.testData.labels);
    },

    compare(cfgA, cfgB) {
        const lossA = this.modelA.trainingHistory[this.modelA.trainingHistory.length - 1];
        const lossB = this.modelB.trainingHistory[this.modelB.trainingHistory.length - 1];
        const timeA = this.modelA.trainingTime;
        const timeB = this.modelB.trainingTime;

        const betterLoss   = lossA  < lossB  ? 'Model A' : lossB  < lossA  ? 'Model B' : 'Tie';
        const betterAcc    = this.accuracyA > this.accuracyB ? 'Model A' : this.accuracyB > this.accuracyA ? 'Model B' : 'Tie';
        const fasterModel  = timeA  < timeB  ? 'Model A' : timeB  < timeA  ? 'Model B' : 'Tie';
        const lossDiff     = Math.abs(lossA - lossB).toFixed(6);
        const accDiff      = Math.abs(this.accuracyA - this.accuracyB);

        const fmtLayers = layers => layers.join(' → ');

        document.getElementById('comparisonText').innerHTML = `
            <div class="insight">
                <strong>Model Configurations</strong><br>
                <table style="width:100%;border-collapse:collapse;margin-top:8px;font-size:0.88em;">
                    <tr style="color:#94a3b8;">
                        <th style="text-align:left;padding:4px 8px;">Setting</th>
                        <th style="text-align:left;padding:4px 8px;">Model A</th>
                        <th style="text-align:left;padding:4px 8px;">Model B</th>
                    </tr>
                    <tr><td style="padding:4px 8px;">Optimizer</td><td style="padding:4px 8px;">${cfgA.optimizer}</td><td style="padding:4px 8px;">${cfgB.optimizer}</td></tr>
                    <tr><td style="padding:4px 8px;">Learning Rate</td><td style="padding:4px 8px;">${cfgA.learningRate}</td><td style="padding:4px 8px;">${cfgB.learningRate}</td></tr>
                    <tr><td style="padding:4px 8px;">Activation</td><td style="padding:4px 8px;">${cfgA.activation}</td><td style="padding:4px 8px;">${cfgB.activation}</td></tr>
                    <tr><td style="padding:4px 8px;">Hidden Layers</td><td style="padding:4px 8px;">${fmtLayers(cfgA.hiddenLayers)}</td><td style="padding:4px 8px;">${fmtLayers(cfgB.hiddenLayers)}</td></tr>
                </table>
            </div>

            <div class="insight">
                <strong>Performance Summary</strong><br>
                <table style="width:100%;border-collapse:collapse;margin-top:8px;font-size:0.88em;">
                    <tr style="color:#94a3b8;">
                        <th style="text-align:left;padding:4px 8px;">Metric</th>
                        <th style="text-align:left;padding:4px 8px;">Model A</th>
                        <th style="text-align:left;padding:4px 8px;">Model B</th>
                        <th style="text-align:left;padding:4px 8px;">Winner</th>
                    </tr>
                    <tr>
                        <td style="padding:4px 8px;">Final Loss</td>
                        <td style="padding:4px 8px;color:${lossA <= lossB ? '#10b981' : '#f87171'}">${lossA.toFixed(6)}</td>
                        <td style="padding:4px 8px;color:${lossB <= lossA ? '#10b981' : '#f87171'}">${lossB.toFixed(6)}</td>
                        <td style="padding:4px 8px;">${betterLoss} ${betterLoss !== 'Tie' ? `(−${lossDiff})` : ''}</td>
                    </tr>
                    <tr>
                        <td style="padding:4px 8px;">Accuracy</td>
                        <td style="padding:4px 8px;color:${this.accuracyA >= this.accuracyB ? '#10b981' : '#f87171'}">${(this.accuracyA * 100).toFixed(2)}%</td>
                        <td style="padding:4px 8px;color:${this.accuracyB >= this.accuracyA ? '#10b981' : '#f87171'}">${(this.accuracyB * 100).toFixed(2)}%</td>
                        <td style="padding:4px 8px;">${betterAcc} ${betterAcc !== 'Tie' ? `(+${(accDiff * 100).toFixed(2)}%)` : ''}</td>
                    </tr>
                    <tr>
                        <td style="padding:4px 8px;">Training Time</td>
                        <td style="padding:4px 8px;color:${timeA <= timeB ? '#10b981' : '#f87171'}">${timeA.toFixed(0)}ms</td>
                        <td style="padding:4px 8px;color:${timeB <= timeA ? '#10b981' : '#f87171'}">${timeB.toFixed(0)}ms</td>
                        <td style="padding:4px 8px;">${fasterModel}</td>
                    </tr>
                </table>
            </div>
        `;
    }
};

document.addEventListener('DOMContentLoaded', () => App.init());
