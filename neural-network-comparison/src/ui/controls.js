/**
 * UI control handlers for hyperparameter adjustment
 */

const Controls = {
    init() {
        // Range input displays
        document.getElementById('datasetSize').addEventListener('input', (e) => {
            document.getElementById('datasetSizeDisplay').textContent = e.target.value;
        });

        document.getElementById('learningRate').addEventListener('input', (e) => {
            document.getElementById('learningRateDisplay').textContent = e.target.value;
        });

        document.getElementById('hiddenSize').addEventListener('input', (e) => {
            document.getElementById('hiddenSizeDisplay').textContent = e.target.value;
        });

        document.getElementById('epochs').addEventListener('input', (e) => {
            document.getElementById('epochsDisplay').textContent = e.target.value;
        });

        // Train button
        document.getElementById('trainButton').addEventListener('click', () => {
            this.handleTrain();
        });

        // Reset button
        document.getElementById('resetButton').addEventListener('click', () => {
            this.handleReset();
        });
    },

    getHyperparameters() {
        return {
            datasetSize: parseInt(document.getElementById('datasetSize').value),
            learningRate: parseFloat(document.getElementById('learningRate').value),
            hiddenSize: parseInt(document.getElementById('hiddenSize').value),
            epochs: parseInt(document.getElementById('epochs').value)
        };
    },

    setButtonState(disabled) {
        document.getElementById('trainButton').disabled = disabled;
        document.getElementById('resetButton').disabled = disabled;
    },

    updateStats(model, type) {
        const suffix = type === 'manual' ? 'Manual' : 'TensorFlow';
        const prefix = type === 'manual' ? 'manual' : 'tf';

        const loss = model.trainingHistory[model.trainingHistory.length - 1];
        const time = model.trainingTime;
        const epochs = model.trainingHistory.length;

        document.getElementById(`${prefix}Loss`).textContent = loss.toFixed(6);
        document.getElementById(`${prefix}Time`).textContent = `${time.toFixed(2)}ms`;
        document.getElementById(`${prefix}Epochs`).textContent = epochs;
    },

    async handleTrain() {
        this.setButtonState(true);
        
        const params = this.getHyperparameters();
        console.log('🚀 Starting training with params:', params);

        try {
            await App.train(params);
        } catch (error) {
            console.error('❌ Training error:', error);
            alert('Training failed: ' + error.message);
        } finally {
            this.setButtonState(false);
        }
    },

    handleReset() {
        // Reset stats display
        const stats = ['Loss', 'Accuracy', 'Time', 'Epochs'];
        const prefixes = ['manual', 'tf'];

        prefixes.forEach(prefix => {
            stats.forEach(stat => {
                const id = prefix + stat;
                if (stat === 'Epochs') {
                    document.getElementById(id).textContent = '0';
                } else {
                    document.getElementById(id).textContent = '—';
                }
            });
        });

        document.getElementById('comparisonText').innerHTML = '';
        console.log('↻ Reset complete');
    }
};
