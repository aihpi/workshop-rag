"""Figure helpers.

Only the pieces that more than one cell needs: the density estimator and the
figure-to-base64 encoder. The plots themselves stay in the notebooks, where the
plotting code is part of what participants read.
"""

import base64
import io

import numpy as np


def gaussian_kde(data, x_grid, bandwidth=None):
    """Gaussian kernel density estimate, using numpy only (no SciPy needed).

    Bandwidth defaults to Silverman's rule of thumb.
    """
    data = np.asarray(data, dtype=float)
    if bandwidth is None:
        bandwidth = 1.06 * np.std(data) * len(data) ** (-1 / 5)
    if bandwidth < 1e-6:
        bandwidth = 0.01
    kernels = np.exp(-0.5 * ((x_grid[:, None] - data[None, :]) / bandwidth) ** 2)
    return kernels.sum(axis=1) / (len(data) * bandwidth * np.sqrt(2 * np.pi))


def fig_to_base64(fig) -> str:
    """Convert a matplotlib figure to a PNG base64 string, closing the figure."""
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')
