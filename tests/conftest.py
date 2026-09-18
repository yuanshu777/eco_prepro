import numpy as np
import pytest


def make_fan_mask(H=240, W=320, apex=(160, 18), radius=205, half_angle_deg=42.0):
    yy, xx = np.mgrid[0:H, 0:W]
    dx, dy = xx - apex[0], yy - apex[1]
    r = np.hypot(dx, dy)
    ang = np.degrees(np.arctan2(dx, dy))  # 0 = straight down
    return (r <= radius) & (np.abs(ang) <= half_angle_deg) & (dy >= 0)


@pytest.fixture
def synthetic_cine():
    """Fan of moving speckle on a black screen with static overlays that mimic a clinical layout."""
    rng = np.random.default_rng(0)
    H, W, T = 240, 320, 40
    fan = make_fan_mask(H, W)
    frames = np.zeros((T, H, W, 3), np.uint8)
    base = rng.integers(20, 200, (H, W)).astype(np.float32)
    for t in range(T):
        speckle = base + rng.normal(0, 25, (H, W))
        speckle[fan & (np.mgrid[0:H, 0:W][0] > 200)] = base[fan & (np.mgrid[0:H, 0:W][0] > 200)]  # static far field
        g = np.clip(speckle, 0, 255).astype(np.uint8) * fan
        f = np.repeat(g[..., None], 3, axis=-1)
        # static text blocks (top-left / top-right), static grey bar (right), timeline (bottom)
        f[8:40, 6:60] = 180
        f[8:20, 250:312] = 180
        f[30:210, 306:311] = np.linspace(255, 0, 180)[:, None, None].astype(np.uint8)
        f[228:230, 20:300, 1] = 200
        f[224:234, 20 + (t * 7) % 280: 22 + (t * 7) % 280] = (255, 255, 0)  # sweeping cursor
        frames[t] = f
    return frames, fan
