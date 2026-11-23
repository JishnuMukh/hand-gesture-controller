"""Build anatomically plausible synthetic 21-point hands for tests.

Units are centimetres; palm faces -z, fingers point +y. Flexion curls the
finger toward the palm.
"""

from __future__ import annotations

import math

import numpy as np

MCP = {"index": (-2.4, 8.5), "middle": (-0.6, 9.0), "ring": (1.1, 8.6), "pinky": (2.6, 7.8)}
SEG = {"index": (4.0, 2.4, 2.0), "middle": (4.5, 2.8, 2.2),
       "ring": (4.2, 2.6, 2.1), "pinky": (3.3, 1.9, 1.8)}
BASE = {"index": 5, "middle": 9, "ring": 13, "pinky": 17}

EXTENDED = (0, 0, 0)
CURLED = (80, 100, 60)
HALF = (40, 45, 20)   # index bent toward the thumb during a pinch

THUMB_CMC = np.array([-1.8, 2.0, 0.0])


def _chain(start, lengths, flex_deg):
    pts, theta, p = [], 0.0, np.array(start, dtype=float)
    for length, f in zip(lengths, flex_deg):
        theta += math.radians(f)
        p = p + length * np.array([0.0, math.cos(theta), -math.sin(theta)])
        pts.append(p)
    return pts


def make_hand(index=EXTENDED, middle=EXTENDED, ring=EXTENDED, pinky=EXTENDED,
              thumb="out", thumb_tip=None) -> np.ndarray:
    """thumb: 'out' (abducted), 'tucked' (across the palm) or 'to' with ``thumb_tip``."""
    p = np.zeros((21, 3))
    p[0] = (0, 0, 0)
    flex = {"index": index, "middle": middle, "ring": ring, "pinky": pinky}
    for name, base in BASE.items():
        mcp = np.array([*MCP[name], 0.0])
        p[base] = mcp
        p[base + 1: base + 4] = _chain(mcp, SEG[name], flex[name])

    p[1] = THUMB_CMC
    if thumb == "out":
        d = np.array([-0.8, 0.6, 0.0])
        p[2], p[3], p[4] = THUMB_CMC + 3.5 * d, THUMB_CMC + 6.5 * d, THUMB_CMC + 9.0 * d
    elif thumb == "tucked":
        p[2], p[3], p[4] = (-2.6, 4.4, -0.8), (-1.2, 5.6, -1.6), (1.0, 5.0, -1.5)
    elif thumb == "to":
        tip = np.asarray(thumb_tip, dtype=float)
        mcp = THUMB_CMC + 3.5 * np.array([-0.8, 0.6, 0.0])
        p[2], p[3], p[4] = mcp, mcp + 0.55 * (tip - mcp), tip
    else:
        raise ValueError(thumb)
    return p


def pinch_hand(gap_cm: float) -> np.ndarray:
    """Index half-bent, other fingers curled, thumb tip ``gap_cm`` from the index tip."""
    base = make_hand(index=HALF, middle=CURLED, ring=CURLED, pinky=CURLED)
    index_tip = base[8]
    direction = np.array([-0.85, -0.5, 0.15])
    direction /= np.linalg.norm(direction)
    return make_hand(index=HALF, middle=CURLED, ring=CURLED, pinky=CURLED,
                     thumb="to", thumb_tip=index_tip + gap_cm * direction)


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    return q * np.sign(np.linalg.det(q))
