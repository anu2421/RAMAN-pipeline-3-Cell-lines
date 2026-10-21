import os
import sys

import numpy as np

rng = np.random.default_rng(0)
EFFECT = {"Control": (1.0, 1.0, 1.0), "E.coli": (1.5, 1.0, 0.88), "S.aureus": (1.2, 1.4, 0.92)}


def gauss(x, c, w, a):
    return a * np.exp(-(x - c) ** 2 / (2 * w * w))


def spectrum(offset, f):
    x = np.arange(435, 1850, 0.9)
    xs = x + offset + rng.normal(0, 0.3)
    y = (gauss(xs, 747, 4, 3 * f[0]) + gauss(xs, 782, 5, 1.3 * f[1]) + gauss(xs, 800, 22, 4)
         + gauss(xs, 852, 6, 0.8) + gauss(xs, 1004, 3, 3) + gauss(xs, 1130, 5, 2 * f[0])
         + gauss(xs, 1450, 12, 5) + gauss(xs, 1660, 15, 4.5 * f[2]) + gauss(xs, 1320, 10, 3)
         + gauss(xs, 1590, 8, 2) + gauss(xs, 606, 8, 3) + gauss(xs, 960, 25, 1.5))
    return x, y * rng.uniform(0.85, 1.15) + 0.002 * (x - 400) + rng.normal(0, 0.08, len(x)) + 5


def write(folder, name, offset, f):
    os.makedirs(folder, exist_ok=True)
    x, y = spectrum(offset, f)
    with open(os.path.join(folder, name), "w") as fh:
        fh.write("RamanShift,Intensity\n")
        fh.writelines(f"{a:.3f},{b:.5f}\n" for a, b in zip(x, y))


def main(root):
    for cell_line, offset in [("A549", 0.0), ("K562", 1.0), ("KB", -2.0)]:
        for group, f in EFFECT.items():
            for r in (1, 2, 3):
                for c in range(5):
                    write(os.path.join(root, "folders", cell_line, group, f"R{r}"), f"R{r}_{c}.csv", offset, f)
                    write(os.path.join(root, "sets", cell_line, f"SET-{r}", f"{group[0]}-{r}"), f"cell{c}.csv", offset, f)
    print(f"mock data written to {os.path.abspath(root)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/mock")
