"""Entry point: play the Snake game yourself with the arrow keys / WASD."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from snakeai.human_game import SPEED, main  # noqa: E402

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Play Snake.")
    p.add_argument("--speed", type=int, default=SPEED, help="Frames per second.")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    a = p.parse_args()
    main(speed=a.speed, w=a.width, h=a.height)
