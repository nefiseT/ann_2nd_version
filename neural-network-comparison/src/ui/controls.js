/**
 * UI control handlers — shared environment + per-model hyperparameters
 */

const Controls = {
    init() {
        // Shared environment sliders
        this._bindSlider('datasetSize', 'datasetSizeDisplay');
        this._bindSlider('noise', 'noiseDisplay');
        this._bindSlider('epochs', 'epochsDisplay');

        // Model A sliders
        this._bindSlider('aLearningRate', 'aLearningRateDisplay');

        // Model B sliders
        this._bindSlider('bLearningRate', 'bLearningRateDisplay');

        document.getElementById('trainButton').addEventListener('click', () => this.handleTrain());
        document.getElementById('resetButton').addEventListener('click', () => this.handleReset());
    },

    _bindSlider(id, displayId) {
        const el = document.getElementById(id);
        const display = document.getElementById(displayId);
        if (el && display) {
            el.addEventListener('input', () => { display.textContent = el.value; });
        }
    },

    getSharedParams() {
        return {
            datasetSize: parseInt(document.getElementById('datasetSize').value),
            noise: parseFloat(document.getElementById('noise').value),
            epochs: parseInt(document.getElementById('epochs').value)
        };
    },

    getModelParams(prefix) {
        const hiddenRaw = document.getElementById(`${prefix}HiddenLayers`).value;
        const hiddenLayers = hiddenRaw
            .split(',')
            .map(s => parseInt(s.trim()))
            .filter(n => !isNaN(n) && n > 0);

        return {
            learningRate: parseFloat(document.getElementById(`${prefix}LearningRate`).value),
            activation: document.getElementById(`${prefix}Activation`).value,
            hiddenLayers: hiddenLayers.length > 0 ? hiddenLayers : [8],
            optimizer: document.getElementById(`${prefix}Optimizer`).value
        };
    },

    setButtonState(disabled) {
        document.getElementById('trainButton').disabled = disabled;
        document.getElementById('resetButton').disabled = disabled;
    },

    updateStats(model, prefix) {
        const loss = model.trainingHistory[model.trainingHistory.length - 1];
        const time = model.trainingTime;
        const epochs = model.trainingHistory.length;

        document.getElementById(`${prefix}Loss`).textContent = loss.toFixed(6);
        document.getElementById(`${prefix}Time`).textContent = `${time.toFixed(0)}ms`;
        document.getElementById(`${prefix}Epochs`).textContent = epochs;
    },

    async handleTrain() {
        this.setButtonState(true);

        const shared = this.getSharedParams();
        const modelA = this.getModelParams('a');
        const modelB = this.getModelParams('b');

        try {
            await App.train({ shared, modelA, modelB });
        } catch (error) {
            console.error('Training error:', error);
            alert('Training failed: ' + error.message);
        } finally {
            this.setButtonState(false);
        }
    },

    handleReset() {
        ['a', 'b'].forEach(prefix => {
            ['Loss', 'Accuracy', 'Time', 'Epochs'].forEach(stat => {
                const el = document.getElementById(`${prefix}${stat}`);
                if (el) el.textContent = stat === 'Epochs' ? '0' : '—';
            });
        });
        document.getElementById('comparisonText').innerHTML = '';
    }
};
