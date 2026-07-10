"""Court geometry constants and tiny shared numerics.

Single source for the ITF court dimensions every stage projects into
(via each match's fitted homography) and the moving-average smoother
the event/ending code applies to image-y before differentiating.
Values are the frozen M2-era constants shared by every t*w twin.
"""

import numpy as np

W_C, L_C = 10.97, 23.77       # doubles court width, court length (m)
NET_Y = L_C / 2               # net line in court-y
CENTER_X = W_C / 2            # center mark in court-x
SINGLES_MARGIN = 1.372        # doubles alley width (m): singles line inset


def moving_average(x, k):
    pad = k // 2
    xp = np.pad(x, pad, mode="edge")
    return np.convolve(xp, np.ones(k) / k, mode="valid")
