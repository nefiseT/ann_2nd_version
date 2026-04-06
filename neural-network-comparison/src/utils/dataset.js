/**
 * Generate synthetic dataset for circle classification
 * Task: Classify if point (x, y) is inside or outside a circle
 */

const Dataset = {
    /**
     * Generate random points and labels.
     * Points are in range [-1, 1]. Label: 1 if inside circle (radius 0.6), 0 if outside.
     * noise adds uniform jitter to coordinates before the label is determined.
     */
    generateCircle(numSamples = 500, radius = 0.6, noise = 0) {
        const data = [];
        const labels = [];

        for (let i = 0; i < numSamples; i++) {
            const x = Math.random() * 2 - 1;
            const y = Math.random() * 2 - 1;

            // Apply noise to coordinates used for labeling
            const nx = x + (Math.random() * 2 - 1) * noise;
            const ny = y + (Math.random() * 2 - 1) * noise;

            const distance = Math.sqrt(nx * nx + ny * ny);
            const label = distance < radius ? 1 : 0;

            data.push([x, y]);
            labels.push([label]);
        }

        return { data, labels };
    },

    /**
     * Split data into training and test sets
     */
    split(data, labels, trainRatio = 0.8) {
        const n = data.length;
        const trainSize = Math.floor(n * trainRatio);

        const indices = Array.from({ length: n }, (_, i) => i);
        for (let i = n - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [indices[i], indices[j]] = [indices[j], indices[i]];
        }

        const trainIndices = indices.slice(0, trainSize);
        const testIndices = indices.slice(trainSize);

        return {
            train: {
                data: trainIndices.map(i => data[i]),
                labels: trainIndices.map(i => labels[i])
            },
            test: {
                data: testIndices.map(i => data[i]),
                labels: testIndices.map(i => labels[i])
            }
        };
    }
};
