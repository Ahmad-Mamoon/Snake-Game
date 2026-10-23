(function () {
    'use strict';
    const { SnakeGame, Agent, QNet, LineChart, DIRS, DIR } = window.SnakeAI;
    const $ = (id) => document.getElementById(id);

    // ---------------------------------------------------------------- settings
    // Defaults mirror the Python version (640x480 board / 20px cells = 32x24).
    const DEFAULTS = {
        speedManual: 10, speedTrain: 60, speedWatch: 15,
        turbo: false, render: true,
        cols: 32, rows: 24, showGrid: false,
        maxGames: 0, lr: 0.001, gamma: 0.9, batchSize: 1000, memory: 100000,
        epsStart: 40, epsMin: 0, exploreGames: 80, frameLimit: 100, hidden: 256,
        rewardFood: 10, rewardDeath: -10, rewardStep: 0,
        autoSave: true,
    };
    // [min, max, integer]
    const LIMITS = {
        speedManual: [1, 1000, true], speedTrain: [1, 1000, true], speedWatch: [1, 1000, true],
        cols: [8, 60, true], rows: [6, 45, true],
        maxGames: [0, 1000000, true], lr: [0.000001, 1, false], gamma: [0, 0.999, false],
        batchSize: [1, 10000, true], memory: [100, 1000000, true],
        epsStart: [0, 100, false], epsMin: [0, 100, false], exploreGames: [0, 1000000, true],
        frameLimit: [0, 10000, true], hidden: [4, 1024, true],
        rewardFood: [-1000, 1000, false], rewardDeath: [-1000, 1000, false], rewardStep: [-1000, 1000, false],
    };
    const KEYS = { settings: 'snakeai.settings.v1', model: 'snakeai.model.v1', best: 'snakeai.best.v1' };
    const MODE_LABEL = { manual: 'Play', train: 'Train AI', watch: 'Watch AI' };
    const SPEED_KEY = { manual: 'speedManual', train: 'speedTrain', watch: 'speedWatch' };

    const store = {
        get(k, fallback) {
            try { const v = localStorage.getItem(k); return v == null ? fallback : JSON.parse(v); } catch { return fallback; }
        },
        set(k, v) {
            try { localStorage.setItem(k, JSON.stringify(v)); return true; } catch { return false; }
        },
    };

    function clampField(key, v) {
        const [min, max, int] = LIMITS[key];
        let c = Math.min(max, Math.max(min, v));
        if (int) c = Math.round(c);
        return c;
    }

    function sanitize(raw) {
        const out = {};
        if (!raw || typeof raw !== 'object') return out;
        for (const k of Object.keys(DEFAULTS)) {
            const v = raw[k];
            if (typeof DEFAULTS[k] === 'boolean' && typeof v === 'boolean') out[k] = v;
            else if (typeof DEFAULTS[k] === 'number' && Number.isFinite(v)) out[k] = clampField(k, v);
        }
        return out;
    }

    let settings = { ...DEFAULTS, ...sanitize(store.get(KEYS.settings, {})) };
    const saveSettings = () => store.set(KEYS.settings, settings);

    const agentCfg = () => ({
        lr: settings.lr, gamma: settings.gamma, hidden: settings.hidden,
        batchSize: settings.batchSize, memory: settings.memory,
        epsStart: settings.epsStart / 100, epsMin: settings.epsMin / 100, exploreGames: settings.exploreGames,
    });
    const gameCfg = (ai) => ({
        cols: settings.cols, rows: settings.rows,
        rewards: { food: settings.rewardFood, death: settings.rewardDeath, step: settings.rewardStep },
        frameLimit: ai ? settings.frameLimit : 0,
    });

    // ---------------------------------------------------------------- state
    let agent = new Agent(agentCfg());
    const history = { scores: [], mean: [], mean100: [], total: 0, sum100: 0, record: 0 };

    const legacyBest = Number(store.get('snake-best', 0)) || 0;
    const modes = {
        manual: { game: new SnakeGame(gameCfg(false)), status: 'idle', queue: [], best: store.get(KEYS.best, legacyBest), newBest: false },
        train: { game: new SnakeGame(gameCfg(true)), status: 'idle', runStart: 0 },
        watch: { game: new SnakeGame(gameCfg(true)), status: 'idle', best: 0, games: 0, total: 0, last: null, restartAt: 0 },
    };
    let mode = 'manual';

    let boardDirty = true, statsDirty = true, chartDirty = true, overlayDirty = true;

    function resetHistory(scores = []) {
        Object.assign(history, { scores: [], mean: [], mean100: [], total: 0, sum100: 0, record: 0 });
        for (const s of scores) pushScore(s);
        chartDirty = statsDirty = true;
    }

    function pushScore(score) {
        const h = history;
        h.scores.push(score);
        h.total += score;
        h.sum100 += score;
        const n = h.scores.length;
        if (n > 100) h.sum100 -= h.scores[n - 101];
        h.mean.push(h.total / n);
        h.mean100.push(h.sum100 / Math.min(n, 100));
        const isRecord = score > h.record;
        if (isRecord) h.record = score;
        return isRecord;
    }

    // ---------------------------------------------------------------- ticks
    function tickManual() {
        const m = modes.manual, g = m.game;
        const r = g.stepDir(m.queue.length ? m.queue.shift() : undefined);
        if (r.done) {
            m.status = 'over';
            m.newBest = g.score > m.best;
            if (m.newBest) { m.best = g.score; store.set(KEYS.best, m.best); }
            overlayDirty = true;
        }
        statsDirty = true;
    }

    function tickTrain() {
        const m = modes.train, g = m.game;
        const state = g.getState();
        const action = agent.act(state, false);
        const r = g.stepAction(action);
        const t = { state, action, reward: r.reward, next: g.getState(), done: r.done };
        agent.trainShort(t);
        agent.remember(t);
        if (r.done) {
            agent.nGames++;
            agent.trainLong();
            if (pushScore(r.score) && settings.autoSave && r.score > 0) saveModel(true);
            g.configure(gameCfg(true));
            g.reset();
            chartDirty = true;
            if (settings.maxGames > 0 && agent.nGames - m.runStart >= settings.maxGames) {
                setStatus('over');
                toast(`Finished ${settings.maxGames} games. Record so far: ${history.record}.`);
            }
        }
        statsDirty = true;
    }

    function tickWatch() {
        const m = modes.watch, g = m.game;
        if (g.done) { g.configure(gameCfg(true)); g.reset(); statsDirty = true; return; }
        const r = g.stepAction(agent.act(g.getState(), true));
        if (r.done) {
            m.games++; m.total += r.score; m.last = r.score;
            if (r.score > m.best) m.best = r.score;
            if (!settings.turbo) m.restartAt = performance.now() + 600;
        }
        statsDirty = true;
    }

    const TICK = { manual: tickManual, train: tickTrain, watch: tickWatch };

    // ---------------------------------------------------------------- loop
    let last = performance.now(), acc = 0, stepCount = 0, spsStart = last, measuredSps = 0;
    let lastStatsAt = 0, lastChartAt = 0;

    function currentSpeed() { return settings[SPEED_KEY[mode]]; }
    const isAI = () => mode !== 'manual';

    function frame(now) {
        const dt = Math.min(now - last, 250);
        last = now;
        const m = modes[mode];
        const tick = TICK[mode];
        let ticked = false;

        if (m.status === 'running') {
            const t0 = performance.now();
            const waiting = mode === 'watch' && now < modes.watch.restartAt;
            if (waiting) {
                acc = 0;
            } else if (isAI() && settings.turbo) {
                do { tick(); stepCount++; }
                while (m.status === 'running' && performance.now() - t0 < 12);
                ticked = true;
            } else {
                acc += (dt / 1000) * currentSpeed();
                while (acc >= 1 && m.status === 'running') {
                    tick(); stepCount++; acc -= 1; ticked = true;
                    if (mode === 'watch' && modes.watch.restartAt > now) { acc = 0; break; }
                    if (performance.now() - t0 > 14) { acc = 0; break; }
                }
            }
        } else {
            acc = 0;
        }

        if (now - spsStart >= 500) {
            measuredSps = (stepCount * 1000) / (now - spsStart);
            stepCount = 0; spsStart = now; statsDirty = true;
        }
        if (ticked || boardDirty) { drawBoard(); boardDirty = false; }
        if (overlayDirty) { renderOverlay(); renderButtons(); overlayDirty = false; }
        if (statsDirty && now - lastStatsAt > 100) { renderStats(); lastStatsAt = now; statsDirty = false; }
        if (chartDirty && now - lastChartAt > 250) {
            chart.setData(history); lastChartAt = now; chartDirty = false;
        }
        requestAnimationFrame(frame);
    }

    function setStatus(status, which = mode) {
        modes[which].status = status;
        if (status === 'running' && which === mode) acc = 1; // move immediately, no dead first beat
        overlayDirty = statsDirty = boardDirty = true;
    }

    function toggleRun() {
        const m = modes[mode];
        if (m.status === 'running') { setStatus('paused'); return; }
        if (mode === 'manual' && m.status === 'over') newGame();
        if (mode === 'train' && m.status !== 'paused') m.runStart = agent.nGames;
        setStatus('running');
    }

    function stepOnce() {
        const m = modes[mode];
        if (!isAI() || m.status === 'running') return;
        if (m.status !== 'paused') {
            if (mode === 'train') m.runStart = agent.nGames;
            setStatus('paused');
        }
        TICK[mode]();
        boardDirty = true;
    }

    function newGame() {
        const m = modes[mode];
        m.game.configure(gameCfg(isAI()));
        m.game.reset();
        if (mode === 'manual') { m.queue = []; m.newBest = false; setStatus('idle'); }
        if (mode === 'watch') m.restartAt = 0;
        boardDirty = statsDirty = overlayDirty = true;
    }

    // ---------------------------------------------------------------- board
    const wrap = $('board-wrap');
    const canvas = $('board');
    const ctx = canvas.getContext('2d');
    const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    const colors = {
        bg: css('--board-bg'), grid: css('--board-grid'), head: css('--snake-head'),
        body: css('--snake-body'), inner: css('--snake-inner'), food: css('--food'),
    };
    new ResizeObserver(() => { boardDirty = true; }).observe(wrap);

    function sizeBoard(g) {
        const ratio = `${g.cols} / ${g.rows}`;
        if (wrap.style.aspectRatio !== ratio) {
            wrap.style.aspectRatio = ratio;
            // Keep the whole board on screen on short, wide displays.
            wrap.style.maxWidth = `max(280px, calc((100vh - 230px) * ${g.cols / g.rows}))`;
        }
        const dpr = window.devicePixelRatio || 1;
        const w = Math.max(1, Math.round(wrap.clientWidth * dpr));
        const h = Math.max(1, Math.round(wrap.clientHeight * dpr));
        if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
    }

    function drawBoard() {
        const m = modes[mode], g = m.game;
        sizeBoard(g);
        const W = canvas.width, H = canvas.height;
        const cw = W / g.cols, ch = H / g.rows;
        ctx.fillStyle = colors.bg;
        ctx.fillRect(0, 0, W, H);
        if (isAI() && !settings.render && m.status === 'running') return;

        if (settings.showGrid) {
            ctx.strokeStyle = colors.grid;
            ctx.lineWidth = 1;
            ctx.beginPath();
            for (let x = 1; x < g.cols; x++) { const px = Math.round(x * cw) + 0.5; ctx.moveTo(px, 0); ctx.lineTo(px, H); }
            for (let y = 1; y < g.rows; y++) { const py = Math.round(y * ch) + 0.5; ctx.moveTo(0, py); ctx.lineTo(W, py); }
            ctx.stroke();
        }

        if (g.food) {
            ctx.fillStyle = colors.food;
            ctx.beginPath();
            ctx.arc((g.food.x + 0.5) * cw, (g.food.y + 0.5) * ch, Math.min(cw, ch) * 0.38, 0, Math.PI * 2);
            ctx.fill();
        }

        const gap = Math.max(1, Math.min(cw, ch) * 0.08);
        const inset = Math.min(cw, ch) * 0.28;
        for (let i = g.snake.length - 1; i >= 0; i--) {
            const s = g.snake[i];
            if (s.x < 0 || s.y < 0 || s.x >= g.cols || s.y >= g.rows) continue;
            const x = s.x * cw, y = s.y * ch;
            ctx.fillStyle = i === 0 ? colors.head : colors.body;
            ctx.fillRect(x + gap, y + gap, cw - 2 * gap, ch - 2 * gap);
            if (i > 0) {
                ctx.fillStyle = colors.inner;
                ctx.fillRect(x + inset, y + inset, cw - 2 * inset, ch - 2 * inset);
            }
        }

        // Eyes show which way the head is facing.
        const h = g.head;
        if (h.x >= 0 && h.y >= 0 && h.x < g.cols && h.y < g.rows) {
            const d = DIRS[g.dir];
            const cx = (h.x + 0.5) * cw, cy = (h.y + 0.5) * ch;
            const fwd = Math.min(cw, ch) * 0.18, side = Math.min(cw, ch) * 0.2, r = Math.max(1, Math.min(cw, ch) * 0.08);
            ctx.fillStyle = colors.bg;
            for (const sgn of [-1, 1]) {
                ctx.beginPath();
                ctx.arc(cx + d.x * fwd - d.y * side * sgn, cy + d.y * fwd + d.x * side * sgn, r, 0, Math.PI * 2);
                ctx.fill();
            }
        }
    }

    // ---------------------------------------------------------------- overlay, buttons, stats
    const overlay = $('overlay'), overlayTitle = $('overlay-title'), overlaySub = $('overlay-sub');
    const touch = matchMedia('(hover: none) and (pointer: coarse)').matches;

    function overlayText() {
        const m = modes[mode];
        const untrained = agent.nGames === 0 && !agent.loaded;
        if (mode === 'manual') {
            if (m.status === 'idle') return ['Ready?', touch ? 'Swipe on the board or tap a direction to start.' : 'Press an arrow key or W A S D to start. Space pauses.'];
            if (m.status === 'paused') return ['Paused', touch ? 'Tap the board or a direction to resume.' : 'Press Space or an arrow key to resume.'];
            if (m.status === 'over') return [`Game over · ${m.game.score}${m.newBest ? ' · new best!' : ''}`, touch ? 'Tap the board to play again.' : 'Press Space or Enter to play again.'];
            return null;
        }
        if (m.status === 'running') {
            return settings.render ? null : ['Training without drawing', 'Turn on “Draw the board” in Speed settings to watch the snake.'];
        }
        if (mode === 'train') {
            if (m.status === 'paused') return ['Training paused', 'Press Start or Space to continue. Settings changes apply right away.'];
            if (m.status === 'over') return [`Finished ${settings.maxGames} games`, 'Press Start to run another batch, or switch to Watch AI to see how it plays.'];
            return ['Train the AI', agent.nGames
                ? `This agent has played ${agent.nGames} games. Press Start to keep training.`
                : 'Press Start and watch it learn. The first games are mostly random moves.'];
        }
        if (m.status === 'paused') return ['Paused', 'Press Start or Space to continue.'];
        return ['Watch the AI', untrained
            ? 'This agent hasn’t been trained yet, so it will play badly. Train it first, or press “Load pretrained”.'
            : 'The agent plays using only what it has learned, with no random moves.'];
    }

    function renderOverlay() {
        const t = overlayText();
        overlay.hidden = !t;
        if (t) { overlayTitle.textContent = t[0]; overlaySub.textContent = t[1]; }
    }

    const btnRun = $('btn-run'), btnStep = $('btn-step'), btnRestart = $('btn-restart'), keyHint = $('key-hint');
    function renderButtons() {
        const m = modes[mode];
        btnRun.textContent = m.status === 'running' ? 'Pause'
            : m.status === 'paused' ? 'Resume'
            : (mode === 'manual' && m.status === 'over') ? 'Play again'
            : mode === 'train' ? 'Start training' : 'Start';
        btnStep.hidden = !isAI();
        btnStep.disabled = m.status === 'running';
        btnRestart.hidden = mode === 'train';
        keyHint.innerHTML = mode === 'manual'
            ? '<kbd>←↑↓→</kbd> / <kbd>WASD</kbd> move · <kbd>Space</kbd> pause'
            : '<kbd>Space</kbd> start / pause';
    }

    function tiles(el, items) {
        const sig = items.map(i => i[0]).join('|');
        if (el.dataset.sig !== sig) {
            el.dataset.sig = sig;
            el.replaceChildren(...items.map(([label]) => {
                const d = document.createElement('div');
                d.className = 'stat';
                d.innerHTML = '<span class="label"></span><span class="value"></span>';
                d.firstChild.textContent = label;
                return d;
            }));
        }
        items.forEach(([, value], i) => {
            const v = el.children[i].lastChild;
            const s = String(value);
            if (v.textContent !== s) v.textContent = s;
        });
    }

    const fmt = (v, d = 2) => (Number.isFinite(v) ? v.toFixed(d) : '–');
    const fmtInt = (v) => Math.round(v).toLocaleString();
    const pct = (v) => `${(v * 100).toFixed(v < 0.1 && v > 0 ? 1 : 0)}%`;

    function speedText() {
        if (isAI() && settings.turbo) return modes[mode].status === 'running' ? `${fmtInt(measuredSps)}/s` : 'Max';
        return `${currentSpeed()}/s`;
    }

    function renderStats() {
        const m = modes[mode], g = m.game;
        if (mode === 'manual') {
            tiles($('stage-stats'), [['Score', g.score], ['Best', m.best], ['Length', g.snake.length], ['Speed', speedText()]]);
        } else if (mode === 'train') {
            const runDone = agent.nGames - m.runStart;
            tiles($('stage-stats'), [
                ['Game', fmtInt(history.scores.length + 1)], ['Score', g.score], ['Record', history.record],
                ['This run', settings.maxGames > 0 ? `${Math.min(runDone, settings.maxGames)} / ${settings.maxGames}` : fmtInt(runDone)],
                ['Steps/s', m.status === 'running' ? fmtInt(measuredSps) : '–'],
            ]);
        } else {
            tiles($('stage-stats'), [
                ['Score', g.score], ['Best', m.best], ['Games', m.games],
                ['Average', m.games ? fmt(m.total / m.games) : '–'], ['Speed', speedText()],
            ]);
        }
        const n = history.scores.length;
        tiles($('train-stats'), [
            ['Games', fmtInt(n)], ['Record', history.record],
            ['Mean', n ? fmt(history.mean[n - 1]) : '–'], ['Last 100', n ? fmt(history.mean100[n - 1]) : '–'],
            ['Exploration', pct(agent.epsilon())], ['Memory', fmtInt(agent.memory.size)],
            ['Loss', n ? fmt(agent.lastLoss, 3) : '–'],
        ]);
        $('speed-out').textContent = speedText();
    }

    const chartColor = (n) => css(n);
    const chart = new LineChart($('chart'), [
        { key: 'scores', color: chartColor('--series-1'), width: 1 },
        { key: 'mean', color: chartColor('--series-2'), width: 2 },
        { key: 'mean100', color: chartColor('--series-3'), width: 2 },
    ]);

    // ---------------------------------------------------------------- toast
    const toastEl = $('toast');
    let toastTimer = 0;
    function toast(msg, isError = false) {
        toastEl.textContent = msg;
        toastEl.classList.toggle('error', isError);
        clearTimeout(toastTimer);
        toastTimer = setTimeout(() => { toastEl.textContent = ''; }, 6000);
    }

    // ---------------------------------------------------------------- mode switching
    const tabs = [...document.querySelectorAll('.mode-tabs [data-mode]')];
    function setMode(next) {
        if (next === mode) return;
        if (modes[mode].status === 'running') modes[mode].status = 'paused';
        mode = next;
        document.body.dataset.mode = mode;
        tabs.forEach(t => t.setAttribute('aria-selected', String(t.dataset.mode === mode)));
        applyPendingBoardSize();
        syncSpeedControl();
        boardDirty = statsDirty = overlayDirty = true;
    }
    tabs.forEach(t => t.addEventListener('click', () => setMode(t.dataset.mode)));

    // Apply a changed board size right away to games that are not mid-play.
    function applyPendingBoardSize() {
        for (const [name, m] of Object.entries(modes)) {
            const g = m.game;
            if (g.cols === settings.cols && g.rows === settings.rows) continue;
            const midGame = g.frame > 0 && !g.done;
            if (midGame && (m.status === 'running' || m.status === 'paused')) continue;
            g.configure(gameCfg(name !== 'manual'));
            g.reset();
            if (name === 'manual' && m.status === 'over') m.status = 'idle';
        }
        boardDirty = overlayDirty = true;
    }

    // ---------------------------------------------------------------- settings UI
    const speedInput = $('speed');
    const toSlider = (sps) => Math.round((Math.log10(sps) / 3) * 100);
    const fromSlider = (p) => clampField('speedManual', Math.round(Math.pow(10, (p / 100) * 3)));

    function syncSpeedControl() {
        speedInput.value = toSlider(currentSpeed());
        const turbo = isAI() && settings.turbo;
        speedInput.disabled = turbo;
        $('speed-mode-label').textContent = `(${MODE_LABEL[mode]})`;
        document.querySelectorAll('[data-ai-only]').forEach(el => el.classList.toggle('is-disabled', !isAI()));
        $('speed-out').textContent = speedText();
    }

    speedInput.addEventListener('input', () => {
        settings[SPEED_KEY[mode]] = fromSlider(Number(speedInput.value));
        saveSettings();
        statsDirty = true;
        $('speed-out').textContent = speedText();
    });
    // A slider left focused after dragging would capture the arrow keys meant for the snake.
    speedInput.addEventListener('pointerup', () => speedInput.blur());

    const fields = [...document.querySelectorAll('[data-field]')];
    function syncInputs() {
        for (const el of fields) {
            const k = el.dataset.field;
            if (el.type === 'checkbox') el.checked = settings[k];
            else el.value = settings[k];
            el.classList.remove('invalid');
            if (LIMITS[k]) { el.min = LIMITS[k][0]; el.max = LIMITS[k][1]; }
        }
        syncSpeedControl();
        $('hidden-note').hidden = settings.hidden === agent.net.nHid;
    }

    for (const el of fields) {
        const key = el.dataset.field;
        el.addEventListener('change', () => {
            if (el.type === 'checkbox') {
                settings[key] = el.checked;
            } else {
                const v = parseFloat(el.value);
                if (!Number.isFinite(v)) {
                    el.value = settings[key];
                    toast('Please enter a number.', true);
                    return;
                }
                const c = clampField(key, v);
                if (c !== v) toast(`Allowed range is ${LIMITS[key][0]} to ${LIMITS[key][1]}, so it was set to ${c}.`);
                settings[key] = c;
                el.value = c;
            }
            onSettingsChanged(key);
        });
        if (el.type === 'number') {
            el.addEventListener('input', () => el.classList.toggle('invalid', !el.checkValidity()));
            el.addEventListener('keydown', (e) => { if (e.key === 'Enter') el.blur(); });
        }
    }

    function onSettingsChanged(key) {
        saveSettings();
        agent.setConfig(agentCfg());
        for (const [name, m] of Object.entries(modes)) m.game.configure(gameCfg(name !== 'manual'));
        if (key === 'cols' || key === 'rows' || key === '*') applyPendingBoardSize();
        $('hidden-note').hidden = settings.hidden === agent.net.nHid;
        if (key === 'turbo' || key === '*') { syncSpeedControl(); acc = 0; }
        boardDirty = statsDirty = overlayDirty = true;
    }

    $('btn-defaults').addEventListener('click', () => {
        settings = { ...DEFAULTS };
        syncInputs();
        onSettingsChanged('*');
        toast('All settings restored to their defaults. Your agent and its training were kept.');
    });

    // ---------------------------------------------------------------- model management
    let autoSaveWarned = false;
    function saveModel(auto) {
        const ok = store.set(KEYS.model, {
            net: agent.net.toJSON(), nGames: agent.nGames, scores: history.scores, savedAt: Date.now(),
        });
        if (!auto) {
            toast(ok ? `Saved in this browser after ${history.scores.length} games (record ${history.record}).`
                : 'Couldn’t save: browser storage is full or blocked.', !ok);
        } else if (ok) {
            toast(`New record: ${history.record}. Model saved.`);
        } else if (!autoSaveWarned) {
            autoSaveWarned = true;
            toast('Auto-save failed: browser storage is full or blocked.', true);
        }
        return ok;
    }

    function confirmReplace() {
        return history.scores.length === 0
            || confirm('Replace the current agent? Any training you haven’t saved will be lost.');
    }

    function installAgent(a, scores = []) {
        agent = a;
        resetHistory(scores);
        modes.train.runStart = agent.nGames;
        if (modes.train.status === 'over') modes.train.status = 'idle';
        for (const name of ['train', 'watch']) { modes[name].game.configure(gameCfg(true)); modes[name].game.reset(); }
        Object.assign(modes.watch, { best: 0, games: 0, total: 0, last: null, restartAt: 0 });
        if (settings.hidden !== a.net.nHid) {
            settings.hidden = a.net.nHid;
            saveSettings();
            syncInputs();
        }
        $('hidden-note').hidden = true;
        boardDirty = statsDirty = overlayDirty = chartDirty = true;
    }

    function agentFromData(data) {
        const net = QNet.fromJSON(data.net || data);
        const a = new Agent({ ...agentCfg(), hidden: net.nHid }, net);
        // Models without a game count (e.g. exported from Python) skip the random-exploration phase.
        a.nGames = Number.isFinite(data.nGames) ? data.nGames : settings.exploreGames;
        a.loaded = true;
        const scores = Array.isArray(data.scores) ? data.scores.filter(Number.isFinite) : [];
        return { a, scores };
    }

    $('btn-save').addEventListener('click', () => saveModel(false));

    $('btn-load').addEventListener('click', () => {
        const data = store.get(KEYS.model, null);
        if (!data) { toast('Nothing saved in this browser yet.', true); return; }
        if (!confirmReplace()) return;
        try {
            const { a, scores } = agentFromData(data);
            installAgent(a, scores);
            toast(`Loaded the saved agent (${scores.length} games, record ${history.record}).`);
        } catch (err) { toast(`Couldn’t load the saved model: ${err.message}`, true); }
    });

    $('btn-pretrained').addEventListener('click', async () => {
        if (!confirmReplace()) return;
        try {
            const res = await fetch('pretrained.json', { cache: 'no-cache' });
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            const { a } = agentFromData(await res.json());
            installAgent(a);
            toast('Loaded the pretrained model from the Python version. Try Watch AI.');
        } catch (err) {
            toast(location.protocol === 'file:'
                ? 'Browsers block loading files from disk. Run “python -m http.server” in the docs folder, or use Upload .json with docs/pretrained.json.'
                : `Couldn’t load pretrained.json (${err.message}).`, true);
        }
    });

    $('btn-export').addEventListener('click', () => {
        const data = { ...agent.net.toJSON(), nGames: agent.nGames, scores: history.scores, exportedAt: new Date().toISOString() };
        const blob = new Blob([JSON.stringify(data)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = Object.assign(document.createElement('a'), { href: url, download: `snake-dqn-${history.scores.length}-games.json` });
        document.body.appendChild(a); a.click(); a.remove();
        setTimeout(() => URL.revokeObjectURL(url), 1000);
    });

    const fileInput = $('file-import');
    $('btn-import').addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', async () => {
        const file = fileInput.files[0];
        fileInput.value = '';
        if (!file || !confirmReplace()) return;
        try {
            const { a, scores } = agentFromData(JSON.parse(await file.text()));
            installAgent(a, scores);
            toast(`Loaded ${file.name}.`);
        } catch (err) { toast(`Couldn’t read ${file.name}: ${err.message}`, true); }
    });

    $('btn-new-agent').addEventListener('click', () => {
        if (!confirmReplace()) return;
        installAgent(new Agent(agentCfg()));
        toast(`Started a fresh agent with ${settings.hidden} hidden neurons.`);
    });

    // ---------------------------------------------------------------- input
    btnRun.addEventListener('click', toggleRun);
    btnStep.addEventListener('click', stepOnce);
    btnRestart.addEventListener('click', newGame);

    const KEY_DIRS = {
        ArrowRight: DIR.RIGHT, d: DIR.RIGHT, ArrowLeft: DIR.LEFT, a: DIR.LEFT,
        ArrowUp: DIR.UP, w: DIR.UP, ArrowDown: DIR.DOWN, s: DIR.DOWN,
    };

    function manualInput(dir) {
        const m = modes.manual;
        if (m.status === 'over') return;
        if (m.game.frame === 0 && !m.queue.length) { m.game.orient(dir); boardDirty = true; }
        const lastDir = m.queue.length ? m.queue[m.queue.length - 1] : m.game.dir;
        if (dir !== lastDir && (dir + 2) % 4 !== lastDir && m.queue.length < 3) m.queue.push(dir);
        if (m.status !== 'running') setStatus('running');
    }

    document.addEventListener('keydown', (e) => {
        if (e.ctrlKey || e.metaKey || e.altKey) return;
        const t = e.target instanceof Element ? e.target : document.body;
        const key = e.key.length === 1 ? e.key.toLowerCase() : e.key;
        if (key === 'Escape') { document.activeElement?.blur?.(); return; }
        if (t.closest('input[type="number"], input[type="text"], select, textarea')) return;
        const isArrow = key.startsWith('Arrow');
        if (isArrow && t.closest('input[type="range"]')) return;
        const activates = t.closest('button, summary, a, input[type="checkbox"]');
        if ((key === ' ' || key === 'Enter') && activates) return; // native keyboard activation

        if (mode === 'manual') {
            const dir = KEY_DIRS[key];
            if (dir !== undefined) { e.preventDefault(); manualInput(dir); return; }
            if (key === ' ' || key === 'Enter' || key === 'p') { e.preventDefault(); toggleRun(); }
        } else if (key === ' ') {
            e.preventDefault();
            toggleRun();
        }
    });

    // After a mouse click, drop focus from controls so Space/arrows go to the game again.
    document.addEventListener('click', (e) => {
        if (e.detail === 0) return; // keyboard activation: keep focus for accessibility
        setTimeout(() => {
            const a = document.activeElement;
            if (a && a.matches('button, summary, input[type="checkbox"], input[type="range"]')) a.blur();
        }, 0);
    });

    // Touch: swipe on the board to steer, tap to start / pause / restart.
    let swipe = null;
    wrap.addEventListener('pointerdown', (e) => {
        wrap.focus({ preventScroll: true });
        swipe = { x: e.clientX, y: e.clientY, used: false };
    });
    wrap.addEventListener('pointermove', (e) => {
        if (!swipe || swipe.used || mode !== 'manual' || e.pointerType === 'mouse') return;
        const dx = e.clientX - swipe.x, dy = e.clientY - swipe.y;
        if (Math.max(Math.abs(dx), Math.abs(dy)) < 24) return;
        swipe.used = true;
        if (modes.manual.status === 'over') newGame();
        manualInput(Math.abs(dx) > Math.abs(dy) ? (dx > 0 ? DIR.RIGHT : DIR.LEFT) : (dy > 0 ? DIR.DOWN : DIR.UP));
    });
    wrap.addEventListener('pointerup', () => {
        if (swipe && !swipe.used && modes[mode].status !== 'running') toggleRun();
        else if (swipe && !swipe.used && mode === 'manual' && touch) toggleRun();
        swipe = null;
    });
    wrap.addEventListener('pointercancel', () => { swipe = null; });

    document.querySelectorAll('#dpad [data-dir]').forEach(btn => {
        btn.addEventListener('pointerdown', (e) => {
            e.preventDefault();
            if (modes.manual.status === 'over') newGame();
            manualInput(Number(btn.dataset.dir));
        });
    });

    document.addEventListener('visibilitychange', () => {
        if (document.hidden && modes.manual.status === 'running') setStatus('paused', 'manual');
    });

    // ---------------------------------------------------------------- boot
    const repo = $('repo-link');
    const path = location.pathname.split('/').filter(Boolean);
    if (location.hostname.endsWith('github.io') && path.length >= 1) {
        repo.href = `https://github.com/${location.hostname.split('.')[0]}/${path[0]}`;
    }

    document.body.dataset.mode = mode;
    syncInputs();
    resetHistory();
    requestAnimationFrame(frame);
})();
