// Dependency-free port of model.py (Linear_QNet + QTrainer) and agent.py (Agent).
(function (NS) {
    'use strict';

    // Weight layout matches PyTorch: W1[j * nIn + i] is linear1.weight[j][i].
    class QNet {
        constructor(nIn, nHid, nOut) {
            this.nIn = nIn; this.nHid = nHid; this.nOut = nOut;
            this.W1 = QNet._init(nHid * nIn, nIn);
            this.b1 = QNet._init(nHid, nIn);
            this.W2 = QNet._init(nOut * nHid, nHid);
            this.b2 = QNet._init(nOut, nHid);
            this.h = new Float32Array(nHid);
            this._resetAdam();
        }

        // PyTorch nn.Linear default: U(-1/sqrt(fan_in), 1/sqrt(fan_in)).
        static _init(n, fanIn) {
            const a = new Float32Array(n), k = 1 / Math.sqrt(fanIn);
            for (let i = 0; i < n; i++) a[i] = (Math.random() * 2 - 1) * k;
            return a;
        }

        _resetAdam() {
            this.params = [this.W1, this.b1, this.W2, this.b2];
            this.grads = this.params.map(p => new Float32Array(p.length));
            this.m = this.params.map(p => new Float32Array(p.length));
            this.v = this.params.map(p => new Float32Array(p.length));
            this.t = 0;
        }

        // Writes hidden activations into `h` and returns Q-values in `out`.
        forward(x, out = new Float32Array(this.nOut), h = this.h) {
            const { nIn, nHid, nOut, W1, b1, W2, b2 } = this;
            for (let j = 0; j < nHid; j++) {
                let s = b1[j];
                const row = j * nIn;
                for (let i = 0; i < nIn; i++) s += W1[row + i] * x[i];
                h[j] = s > 0 ? s : 0;
            }
            for (let o = 0; o < nOut; o++) {
                let s = b2[o];
                const row = o * nHid;
                for (let j = 0; j < nHid; j++) s += W2[row + j] * h[j];
                out[o] = s;
            }
            return out;
        }

        maxQ(x) {
            const q = this.forward(x, this._q || (this._q = new Float32Array(this.nOut)), this._h2 || (this._h2 = new Float32Array(this.nHid)));
            let m = q[0];
            for (let o = 1; o < q.length; o++) if (q[o] > m) m = q[o];
            return m;
        }

        argmax(x) {
            const q = this.forward(x, this._q || (this._q = new Float32Array(this.nOut)), this._h2 || (this._h2 = new Float32Array(this.nHid)));
            let best = 0;
            for (let o = 1; o < q.length; o++) if (q[o] > q[best]) best = o;
            return best;
        }

        // One QTrainer.train_step over a batch of transitions:
        // target = r (+ gamma * max Q(s') if not done), MSE over all N*nOut outputs.
        trainBatch(batch, gamma, lr) {
            const { nIn, nHid, nOut, W2 } = this;
            const [gW1, gb1, gW2, gb2] = this.grads;
            gW1.fill(0); gb1.fill(0); gW2.fill(0); gb2.fill(0);

            const N = batch.length;
            const scale = 2 / (N * nOut);
            const h = new Float32Array(nHid), q = new Float32Array(nOut);
            let loss = 0;

            // Targets use the pre-update network for every sample, as PyTorch does.
            const targets = new Float32Array(N);
            for (let n = 0; n < N; n++) {
                const tr = batch[n];
                targets[n] = tr.done ? tr.reward : tr.reward + gamma * this.maxQ(tr.next);
            }

            for (let n = 0; n < N; n++) {
                const tr = batch[n];
                this.forward(tr.state, q, h);
                const a = tr.action;
                const err = q[a] - targets[n];
                loss += err * err;
                const dq = scale * err;

                gb2[a] += dq;
                const row2 = a * nHid;
                for (let j = 0; j < nHid; j++) {
                    if (h[j] <= 0) continue;
                    gW2[row2 + j] += dq * h[j];
                    const dh = dq * W2[row2 + j];
                    gb1[j] += dh;
                    const row1 = j * nIn;
                    for (let i = 0; i < nIn; i++) {
                        const xi = tr.state[i];
                        if (xi !== 0) gW1[row1 + i] += dh * xi;
                    }
                }
            }
            this._adamStep(lr);
            return loss / (N * nOut);
        }

        _adamStep(lr, b1 = 0.9, b2 = 0.999, eps = 1e-8) {
            this.t++;
            const c1 = 1 - Math.pow(b1, this.t), c2 = 1 - Math.pow(b2, this.t);
            for (let k = 0; k < this.params.length; k++) {
                const p = this.params[k], g = this.grads[k], m = this.m[k], v = this.v[k];
                for (let i = 0; i < p.length; i++) {
                    const gi = g[i];
                    m[i] = b1 * m[i] + (1 - b1) * gi;
                    v[i] = b2 * v[i] + (1 - b2) * gi * gi;
                    p[i] -= lr * (m[i] / c1) / (Math.sqrt(v[i] / c2) + eps);
                }
            }
        }

        toJSON() {
            const r = (a) => Array.from(a, v => Math.round(v * 1e6) / 1e6);
            return { format: 'snake-dqn-v1', nIn: this.nIn, nHid: this.nHid, nOut: this.nOut,
                W1: r(this.W1), b1: r(this.b1), W2: r(this.W2), b2: r(this.b2) };
        }

        static fromJSON(d) {
            if (!d || d.nIn !== 11 || d.nOut !== 3 || !d.nHid) throw new Error('Not a Snake DQN model file.');
            const net = new QNet(d.nIn, d.nHid, d.nOut);
            const copy = (dst, src, name) => {
                if (!src || src.length !== dst.length) throw new Error(`Bad weight array "${name}".`);
                dst.set(src);
            };
            copy(net.W1, d.W1, 'W1'); copy(net.b1, d.b1, 'b1');
            copy(net.W2, d.W2, 'W2'); copy(net.b2, d.b2, 'b2');
            return net;
        }
    }

    class ReplayMemory {
        constructor(capacity) { this.buf = []; this.capacity = capacity; this.next = 0; }
        get size() { return this.buf.length; }
        push(t) {
            if (this.buf.length < this.capacity) this.buf.push(t);
            else { this.buf[this.next] = t; this.next = (this.next + 1) % this.capacity; }
        }
        resize(capacity) {
            if (capacity < this.buf.length) {
                // Keep the most recent `capacity` transitions.
                const ordered = this.buf.slice(this.next).concat(this.buf.slice(0, this.next));
                this.buf = ordered.slice(ordered.length - capacity);
                this.next = 0;
            } else if (this.buf.length === this.capacity && this.next !== 0) {
                this.buf = this.buf.slice(this.next).concat(this.buf.slice(0, this.next));
                this.next = 0;
            }
            this.capacity = capacity;
        }
        sample(n) {
            if (this.buf.length <= n) return this.buf;
            // Partial Fisher-Yates on an index array: sampling without replacement like random.sample.
            const idx = this._idx && this._idx.length === this.buf.length ? this._idx : (this._idx = Uint32Array.from({ length: this.buf.length }, (_, i) => i));
            const out = new Array(n);
            for (let i = 0; i < n; i++) {
                const j = i + Math.floor(Math.random() * (idx.length - i));
                const tmp = idx[i]; idx[i] = idx[j]; idx[j] = tmp;
                out[i] = this.buf[idx[i]];
            }
            return out;
        }
        clear() { this.buf = []; this.next = 0; }
    }

    // cfg: {lr, gamma, hidden, batchSize, memory, epsStart (0-1), epsMin (0-1), exploreGames}
    class Agent {
        constructor(cfg, net) {
            this.cfg = { ...cfg };
            this.net = net || new QNet(11, cfg.hidden, 3);
            this.memory = new ReplayMemory(cfg.memory);
            this.nGames = 0;
            this.lastLoss = 0;
        }

        setConfig(cfg) {
            this.cfg = { ...this.cfg, ...cfg, hidden: this.net.nHid };
            if (this.memory.capacity !== this.cfg.memory) this.memory.resize(this.cfg.memory);
        }

        // Same schedule as agent.py at defaults: (80 - n) / 200 ≈ 0.4 * (1 - n / 80).
        epsilon() {
            const { epsStart, epsMin, exploreGames } = this.cfg;
            const decayed = exploreGames > 0 ? epsStart * Math.max(0, 1 - this.nGames / exploreGames) : 0;
            return Math.max(epsMin, decayed);
        }

        act(state, greedy) {
            if (!greedy && Math.random() < this.epsilon()) return Math.floor(Math.random() * 3);
            return this.net.argmax(state);
        }

        trainShort(t) { this.lastLoss = this.net.trainBatch([t], this.cfg.gamma, this.cfg.lr); }
        remember(t) { this.memory.push(t); }
        trainLong() {
            if (!this.memory.size) return;
            this.lastLoss = this.net.trainBatch(this.memory.sample(this.cfg.batchSize), this.cfg.gamma, this.cfg.lr);
        }
    }

    NS.QNet = QNet;
    NS.Agent = Agent;
})(window.SnakeAI = window.SnakeAI || {});
