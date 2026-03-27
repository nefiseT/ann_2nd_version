/**
 * Performance evaluation functions
 */

const Metrics = {
    /**
     * Binary classification accuracy
     * threshold: 0.5 (standard for sigmoid output)
     */
    accuracy(predictions, labels, threshold = 0.5) {
        let correct = 0;
        for (let i = 0; i < predictions.length; i++) {
            const pred = predictions[i][0] > threshold ? 1 : 0;
            const actual = labels[i][0];
            if (pred === actual) correct++;
        }
        return correct / predictions.length;
    },

    /**
     * Mean Squared Error
     */
    mse(predictions, labels) {
        let sum = 0;
        for (let i = 0; i < predictions.length; i++) {
            const diff = predictions[i][0] - labels[i][0];
            sum += diff * diff;
        }
        return sum / predictions.length;
    },

    /**
     * Precision: TP / (TP + FP)
     */
    precision(predictions, labels, threshold = 0.5) {
        let tp = 0, fp = 0;
        for (let i = 0; i < predictions.length; i++) {
            const pred = predictions[i][0] > threshold ? 1 : 0;
            const actual = labels[i][0];
            if (pred === 1 && actual === 1) tp++;
            if (pred === 1 && actual === 0) fp++;
        }
        return tp / (tp + fp) || 0;
    },

    /**
     * Recall: TP / (TP + FN)
     */
    recall(predictions, labels, threshold = 0.5) {
        let tp = 0, fn = 0;
        for (let i = 0; i < predictions.length; i++) {
            const pred = predictions[i][0] > threshold ? 1 : 0;
            const actual = labels[i][0];
            if (pred === 1 && actual === 1) tp++;
            if (pred === 0 && actual === 1) fn++;
        }
        return tp / (tp + fn) || 0;
    }
};
