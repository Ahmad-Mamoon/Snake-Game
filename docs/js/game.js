// Grid-based Snake engine shared by manual play, training and watching.
// Mirrors src/snakeai/game.py: same state features, rewards and frame limit.
(function (NS) {
    'use strict';

    // Clockwise order, so "turn right" = index + 1 and "turn left" = index - 1.
    const DIRS = [
        { name: 'RIGHT', x: 1, y: 0 },
        { name: 'DOWN', x: 0, y: 1 },
        { name: 'LEFT', x: -1, y: 0 },
        { name: 'UP', x: 0, y: -1 },
    ];
    const DIR = { RIGHT: 0, DOWN: 1, LEFT: 2, UP: 3 };

    class SnakeGame {
        constructor(opts) {
            this.configure(opts);
            this.reset();
        }

        // rewards: {food, death, step}; frameLimit: 0 disables the anti-loop timeout.
        configure({ cols, rows, rewards, frameLimit }) {
            if (cols) this.nextCols = cols;
            if (rows) this.nextRows = rows;
            if (rewards) this.rewards = { ...rewards };
            if (frameLimit !== undefined) this.frameLimit = frameLimit;
        }

        reset(dir = Math.floor(Math.random() * 4)) {
            this.cols = this.nextCols;
            this.rows = this.nextRows;
            this.occ = new Uint16Array(this.cols * this.rows);
            this.dir = dir;
            const d = DIRS[this.dir];
            const cx = Math.floor(this.cols / 2), cy = Math.floor(this.rows / 2);
            this.snake = [];
            for (let i = 0; i < 3; i++) this._push({ x: cx - i * d.x, y: cy - i * d.y });
            this.score = 0;
            this.frame = 0;
            this.done = false;
            this.won = false;
            this._placeFood();
        }

        get head() { return this.snake[0]; }

        // Before the first move, turn the snake to face `dir` (body trailing behind it).
        orient(dir) {
            if (this.frame !== 0 || dir === this.dir) return;
            const food = this.food;
            this.reset(dir);
            if (food && this.occ[food.y * this.cols + food.x] === 0) this.food = food;
        }

        _push(p) { this.snake.push(p); this.occ[p.y * this.cols + p.x]++; }
        _unshift(p) {
            this.snake.unshift(p);
            if (this._inBounds(p)) this.occ[p.y * this.cols + p.x]++;
        }
        _pop() { const p = this.snake.pop(); this.occ[p.y * this.cols + p.x]--; }
        _inBounds(p) { return p.x >= 0 && p.y >= 0 && p.x < this.cols && p.y < this.rows; }

        _placeFood() {
            const total = this.cols * this.rows;
            if (this.snake.length >= total) { this.food = null; return; }
            for (let tries = 0; tries < 64; tries++) {
                const i = Math.floor(Math.random() * total);
                if (this.occ[i] === 0) { this.food = { x: i % this.cols, y: Math.floor(i / this.cols) }; return; }
            }
            const free = [];
            for (let i = 0; i < total; i++) if (this.occ[i] === 0) free.push(i);
            const i = free[Math.floor(Math.random() * free.length)];
            this.food = { x: i % this.cols, y: Math.floor(i / this.cols) };
        }

        // Same semantics as Python's `pt in snake[1:]` plus wall check.
        isCollision(p) {
            if (!this._inBounds(p)) return true;
            const h = this.head;
            const own = (p.x === h.x && p.y === h.y) ? 1 : 0;
            return this.occ[p.y * this.cols + p.x] - own > 0;
        }

        // Absolute move: dirIndex in DIRS. Reversal into the neck is ignored.
        stepDir(dirIndex) {
            if (dirIndex !== undefined && (dirIndex + 2) % 4 !== this.dir) this.dir = dirIndex;
            return this._advance();
        }

        // Relative move: 0 = straight, 1 = turn right, 2 = turn left (Python's [1,0,0] etc).
        stepAction(action) {
            if (action === 1) this.dir = (this.dir + 1) % 4;
            else if (action === 2) this.dir = (this.dir + 3) % 4;
            return this._advance();
        }

        _advance() {
            this.frame++;
            const d = DIRS[this.dir];
            const nh = { x: this.head.x + d.x, y: this.head.y + d.y };
            this._unshift(nh);

            const timedOut = this.frameLimit > 0 && this.frame > this.frameLimit * this.snake.length;
            if (this.isCollision(nh) || timedOut) {
                this.done = true;
                this.timedOut = timedOut;
                return { reward: this.rewards.death, done: true, score: this.score };
            }
            if (this.food && nh.x === this.food.x && nh.y === this.food.y) {
                this.score++;
                this._placeFood();
                if (!this.food) { this.done = true; this.won = true; }
                return { reward: this.rewards.food, done: this.done, score: this.score, ate: true };
            }
            this._pop();
            return { reward: this.rewards.step, done: false, score: this.score };
        }

        // 11 features, same order as Agent.get_state in agent.py.
        getState() {
            const h = this.head;
            const pl = { x: h.x - 1, y: h.y }, pr = { x: h.x + 1, y: h.y };
            const pu = { x: h.x, y: h.y - 1 }, pd = { x: h.x, y: h.y + 1 };
            const dl = this.dir === DIR.LEFT, dr = this.dir === DIR.RIGHT;
            const du = this.dir === DIR.UP, dd = this.dir === DIR.DOWN;
            const c = (p) => this.isCollision(p);
            const f = this.food || h;
            const s = new Float32Array(11);
            s[0] = +((dr && c(pr)) || (dl && c(pl)) || (du && c(pu)) || (dd && c(pd)));
            s[1] = +((du && c(pr)) || (dd && c(pl)) || (dl && c(pu)) || (dr && c(pd)));
            s[2] = +((dd && c(pr)) || (du && c(pl)) || (dr && c(pu)) || (dl && c(pd)));
            s[3] = +dl; s[4] = +dr; s[5] = +du; s[6] = +dd;
            s[7] = +(f.x < h.x); s[8] = +(f.x > h.x);
            s[9] = +(f.y < h.y); s[10] = +(f.y > h.y);
            return s;
        }
    }

    NS.DIRS = DIRS;
    NS.DIR = DIR;
    NS.SnakeGame = SnakeGame;
})(window.SnakeAI = window.SnakeAI || {});
