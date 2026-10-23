# Snake AI – Deep-Q-Learning

A reinforcement-learning agent that teaches itself to play Snake, built with
PyTorch and Pygame. It comes with a **browser app** where you can play the game
yourself, or train and watch the AI live, adjusting every setting while it runs.

**Live app:** `https://<your-username>.github.io/Snake-Game/` (see
[Deploy to GitHub Pages](#deploy-to-github-pages)).

## The web app (`docs/`)

Everything runs in the browser and nothing needs to be installed. The neural network
is a dependency-free JavaScript port of the Python model, and it produces the same
outputs as PyTorch for the same weights.

| Mode | What it does |
|------|--------------|
| **Play** | Classic Snake. Arrow keys or WASD to move, Space to pause. Swipe or use the on-screen pad on phones. |
| **Train AI** | Trains a fresh Deep-Q agent live, with a chart of score, mean and last-100 mean. |
| **Watch AI** | The current agent plays with no random moves, so you can see what it has learned. |

Settings, all of which apply immediately (even mid-run):

- **Speed**: a separate speed for each mode (1–1000 steps/s), a *Max speed* mode, and an option to stop drawing the board for faster training
- **Board**: width and height in cells (applied from the next game), and grid lines
- **Training**: games per run, learning rate, gamma, batch size, replay memory, exploration start / min / decay, frame limit, hidden-layer size
- **Rewards**: food, death, and per-step reward
- **Model**: save to or load from the browser, auto-save on each new record, download / upload `.json`, and *Load pretrained* (the weights trained with the Python version)

Settings and your best manual score are remembered in the browser.

## Project layout

```
Snake-Game/
├── src/snakeai/               # Python package
│   ├── agent.py               # DQN agent + training loop
│   ├── game.py                # Pygame environment the AI drives
│   ├── human_game.py          # Keyboard-controlled Snake
│   ├── model.py               # Linear_QNet + QTrainer
│   └── plot.py                # Live training-score plot
├── train.py                   # Train the agent (CLI options below)
├── play.py                    # Play with arrow keys / WASD
├── export_model.py            # model/model.pth -> docs/pretrained.json
├── docs/                      # Web app (GitHub Pages)
│   ├── index.html, style.css
│   ├── pretrained.json        # exported weights for "Load pretrained"
│   └── js/
│       ├── game.js            # Snake engine (same rules/state as game.py)
│       ├── dqn.js             # Q-network, Adam, replay memory, agent
│       ├── chart.js           # training chart
│       └── app.js             # UI, modes, settings, input
├── assets/arial.ttf
├── model/                     # saved weights + plot (git-ignored)
└── requirements.txt
```

## Python version

```bash
pip install -r requirements.txt

python play.py                     # play (R restart, P pause, Esc quit)
python play.py --speed 20          # faster snake

python train.py                    # train with a pygame window + live plot
python train.py --games 300 --no-render --no-plot   # fast headless training
python train.py --resume           # continue from model/model.pth
python train.py --lr 0.0005 --gamma 0.95 --hidden 512 --explore-games 120
python train.py --help             # all options
```

The best model is saved to `model/model.pth` whenever a new record is set.
With `--resume`, the saved model is only replaced when its record is beaten.
To use your own trained model on the website:

```bash
python export_model.py             # writes docs/pretrained.json
```

## How the agent works

- **State (11 features)**: danger straight / right / left, current direction (4 flags), and food left / right / up / down.
- **Network**: `Linear_QNet(11 → 256 → 3)`, giving Q-values for go straight, turn right, and turn left.
- **Rewards**: `+10` for food, `-10` for dying or for taking more than `100 × length` steps without dying.
- **Exploration**: ε-greedy, starting at about 40% random moves and decaying to zero over the first 80 games.
- **Learning**: one update after every step, plus a replay batch (up to 1 000 of the last 100 000 moves) after every game, using Adam with lr 0.001 and γ 0.9.


## License

MIT
