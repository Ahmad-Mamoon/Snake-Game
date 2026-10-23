"""Entry point: train the reinforcement-learning Snake agent."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from snakeai import agent as ag  # noqa: E402
from snakeai.game import SPEED  # noqa: E402


def main():
    p = argparse.ArgumentParser(description="Train the Snake AI agent.")
    p.add_argument("--games", type=int, default=None,
                   help="Stop after N games (default: run until you close the window).")
    p.add_argument("--speed", type=int, default=SPEED, help="Frames per second.")
    p.add_argument("--no-render", action="store_true",
                   help="Don't draw the board (much faster training).")
    p.add_argument("--no-plot", action="store_true",
                   help="Disable the live matplotlib training plot.")
    p.add_argument("--resume", action="store_true",
                   help="Continue training from model/model.pth.")
    p.add_argument("--lr", type=float, default=ag.LR, help="Learning rate.")
    p.add_argument("--gamma", type=float, default=ag.GAMMA, help="Discount factor.")
    p.add_argument("--hidden", type=int, default=ag.HIDDEN_SIZE, help="Hidden layer size.")
    p.add_argument("--batch-size", type=int, default=ag.BATCH_SIZE)
    p.add_argument("--memory", type=int, default=ag.MAX_MEMORY, help="Replay memory size.")
    p.add_argument("--explore-games", type=int, default=ag.EXPLORE_GAMES,
                   help="Games over which random exploration decays to zero.")
    a = p.parse_args()

    ag.train(max_games=a.games, show_plot=not a.no_plot, speed=a.speed,
             render=not a.no_render, resume=a.resume, lr=a.lr, gamma=a.gamma,
             hidden_size=a.hidden, batch_size=a.batch_size, max_memory=a.memory,
             explore_games=a.explore_games)


if __name__ == "__main__":
    main()
