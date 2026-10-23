// Minimal responsive line chart for training history.
(function (NS) {
    'use strict';

    class LineChart {
        constructor(canvas, series) {
            this.canvas = canvas;
            this.ctx = canvas.getContext('2d');
            this.series = series; // [{key, label, color, width}]
            this.data = {};
            new ResizeObserver(() => this.draw()).observe(canvas);
        }

        setData(data) { this.data = data; this.draw(); }

        draw() {
            const { canvas, ctx } = this;
            const dpr = window.devicePixelRatio || 1;
            const cssW = canvas.clientWidth, cssH = canvas.clientHeight;
            if (!cssW || !cssH) return;
            if (canvas.width !== Math.round(cssW * dpr) || canvas.height !== Math.round(cssH * dpr)) {
                canvas.width = Math.round(cssW * dpr);
                canvas.height = Math.round(cssH * dpr);
            }
            ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
            ctx.clearRect(0, 0, cssW, cssH);

            const styles = getComputedStyle(canvas);
            const grid = styles.getPropertyValue('--chart-grid').trim() || '#2a3150';
            const text = styles.getPropertyValue('--chart-text').trim() || '#8a93b4';
            const pad = { l: 36, r: 12, t: 10, b: 24 };
            const w = cssW - pad.l - pad.r, h = cssH - pad.t - pad.b;

            const first = this.data[this.series[0].key] || [];
            const n = first.length;
            let maxY = 1;
            for (const s of this.series) for (const v of (this.data[s.key] || [])) if (v > maxY) maxY = v;
            maxY = niceCeil(maxY);

            ctx.font = '11px ui-sans-serif, system-ui, sans-serif';
            ctx.fillStyle = text;
            ctx.strokeStyle = grid;
            ctx.lineWidth = 1;
            ctx.textAlign = 'right';
            ctx.textBaseline = 'middle';
            for (let i = 0; i <= 4; i++) {
                const v = (maxY * i) / 4, y = pad.t + h - (h * i) / 4;
                ctx.beginPath(); ctx.moveTo(pad.l, y + 0.5); ctx.lineTo(pad.l + w, y + 0.5); ctx.stroke();
                ctx.fillText(formatNum(v), pad.l - 6, y);
            }
            ctx.textAlign = 'center';
            ctx.textBaseline = 'top';
            const xTicks = Math.min(5, Math.max(1, n - 1));
            for (let i = 0; i <= xTicks; i++) {
                const g = Math.round(((n - 1) * i) / xTicks) + 1;
                if (n === 0) break;
                const x = pad.l + (n > 1 ? (w * (g - 1)) / (n - 1) : 0);
                ctx.fillText(String(g), x, pad.t + h + 6);
            }

            if (n === 0) {
                ctx.textBaseline = 'middle';
                ctx.fillText('No games played yet', pad.l + w / 2, pad.t + h / 2);
                return;
            }

            // Downsample to at most ~2 points per pixel to keep drawing cheap on long runs.
            const step = Math.max(1, Math.floor(n / (w * 2)));
            for (const s of this.series) {
                const arr = this.data[s.key] || [];
                ctx.strokeStyle = s.color;
                ctx.lineWidth = s.width || 1.5;
                ctx.lineJoin = 'round';
                ctx.beginPath();
                for (let i = 0; i < arr.length; i += step) {
                    const x = pad.l + (n > 1 ? (w * i) / (n - 1) : w / 2);
                    const y = pad.t + h - (h * arr[i]) / maxY;
                    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
                }
                const li = arr.length - 1;
                ctx.lineTo(pad.l + (n > 1 ? (w * li) / (n - 1) : w / 2), pad.t + h - (h * arr[li]) / maxY);
                ctx.stroke();
            }
        }
    }

    function niceCeil(v) {
        const p = Math.pow(10, Math.floor(Math.log10(v)));
        for (const m of [1, 2, 2.5, 5, 10]) if (m * p >= v) return m * p;
        return 10 * p;
    }
    function formatNum(v) { return Number.isInteger(v) ? String(v) : v.toFixed(1); }

    NS.LineChart = LineChart;
})(window.SnakeAI = window.SnakeAI || {});
