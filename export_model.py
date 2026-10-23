"""Export model/model.pth to docs/pretrained.json so the web app can load it."""
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent


def main(src=ROOT / "model" / "model.pth", dst=ROOT / "docs" / "pretrained.json"):
    sd = torch.load(src, map_location="cpu")
    w1, b1 = sd["linear1.weight"], sd["linear1.bias"]
    w2, b2 = sd["linear2.weight"], sd["linear2.bias"]

    def flat(t):
        return [round(v, 6) for v in t.flatten().tolist()]

    data = {
        "format": "snake-dqn-v1",
        "nIn": w1.shape[1], "nHid": w1.shape[0], "nOut": w2.shape[0],
        "W1": flat(w1), "b1": flat(b1), "W2": flat(w2), "b2": flat(b2),
        "source": "python",
    }
    dst.write_text(json.dumps(data, separators=(",", ":")))
    print(f"Wrote {dst} ({dst.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main(*map(Path, sys.argv[1:3]))
