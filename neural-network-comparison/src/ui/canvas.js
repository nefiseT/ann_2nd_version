/**
 * Visualization utilities for drawing predictions and data
 */

const Canvas = {
    /**
     * Draw decision boundary and training data on canvas
     */
    draw(canvasId, predictions, dataPoints, labels, title = "") {
        const canvas = document.getElementById(canvasId);
        if (!canvas) {
            console.error(`Canvas element with ID '${canvasId}' not found`);
            return;
        }
        const ctx = canvas.getContext('2d');
        if (!ctx) {
            console.error(`Cannot get 2D context for canvas '${canvasId}'`);
            return;
        }
        const width = canvas.width;
        const height = canvas.height;

        // Clear canvas
        ctx.fillStyle = '#0f172a';
        ctx.fillRect(0, 0, width, height);

        // Draw decision boundary as heatmap
        this.drawHeatmap(ctx, predictions, width, height);

        // Draw training points
        this.drawPoints(ctx, dataPoints, labels, width, height);

        // Draw circle (true boundary)
        this.drawCircle(ctx, width, height, 0.6);

        // Draw grid
        this.drawGrid(ctx, width, height);
    },

    /**
     * Draw heatmap of model predictions
     */
    drawHeatmap(ctx, predictions, width, height) {
        const resolution = predictions.length;
        const pixelSize = width / resolution;

        for (let i = 0; i < resolution; i++) {
            for (let j = 0; j < resolution; j++) {
                const value = predictions[i][j];
                
                // Color gradient: blue (0) → red (1)
                const hue = (1 - value) * 240; // 240 = blue, 0 = red
                ctx.fillStyle = `hsl(${hue}, 100%, 50%)`;
                
                ctx.fillRect(j * pixelSize, i * pixelSize, pixelSize, pixelSize);
            }
        }
    },

    /**
     * Draw training data points
     */
    drawPoints(ctx, dataPoints, labels, width, height) {
        const radius = 3;
        const centerX = width / 2;
        const centerY = height / 2;
        const scale = width / 2;

        for (let i = 0; i < dataPoints.length; i++) {
            const [x, y] = dataPoints[i];
            const label = labels[i][0];

            // Convert from [-1, 1] to canvas coordinates
            const canvasX = centerX + x * scale;
            const canvasY = centerY + y * scale;

            // Color by label: green (inside) or red (outside)
            ctx.fillStyle = label === 1 ? '#10b981' : '#ef4444';
            ctx.beginPath();
            ctx.arc(canvasX, canvasY, radius, 0, Math.PI * 2);
            ctx.fill();

            // White border for contrast
            ctx.strokeStyle = '#ffffff';
            ctx.lineWidth = 0.5;
            ctx.stroke();
        }
    },

    /**
     * Draw the true circle boundary
     */
    drawCircle(ctx, width, height, radius) {
        const centerX = width / 2;
        const centerY = height / 2;
        const scale = width / 2;

        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 2;
        ctx.setLineDash([5, 5]);
        ctx.beginPath();
        ctx.arc(centerX, centerY, radius * scale, 0, Math.PI * 2);
        ctx.stroke();
        ctx.setLineDash([]);
    },

    /**
     * Draw coordinate grid and axes
     */
    drawGrid(ctx, width, height) {
        ctx.strokeStyle = '#334155';
        ctx.lineWidth = 0.5;

        const centerX = width / 2;
        const centerY = height / 2;

// Vertical line (x = 0)
        ctx.beginPath();
        ctx.moveTo(centerX, 0);
        ctx.lineTo(centerX, height);
        ctx.stroke();

        // Horizontal line (y = 0)
        ctx.beginPath();
        ctx.moveTo(0, centerY);
        ctx.lineTo(width, centerY);
        ctx.stroke();

        // Grid lines
        const gridSize = 20;
        for (let i = 0; i <= width; i += gridSize) {
            // Vertical grids
            ctx.beginPath();
            ctx.moveTo(i, 0);
            ctx.lineTo(i, height);
            ctx.stroke();

            // Horizontal grids
            ctx.beginPath();
            ctx.moveTo(0, i);
            ctx.lineTo(width, i);
            ctx.stroke();
        }
    }
};
