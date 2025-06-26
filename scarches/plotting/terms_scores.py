from itertools import product
from typing import Union
import numpy as np
import matplotlib.pyplot as plt

def plot_abs_bfs_key(
    scores,
    terms,
    key,
    n_points: int = 30,       # ← made explicit here
    lim_val: float = 2.3,
    fontsize: int = 8,
    scale_y: float = 2,
    yt_step: float = 0.3,
    title: str = None,
    ax=None
):
    txt_args = dict(
        rotation='vertical',
        verticalalignment='bottom',
        horizontalalignment='center',
        fontsize=fontsize,
    )

    ax = ax if ax is not None else plt.axes()
    ax.grid(False)

    bfs = np.abs(scores[key]['bf'])
    srt = np.argsort(bfs)[::-1][:n_points]
    top = bfs.max()

    ax.set_ylim(top=top * scale_y)
    yt = np.arange(0, top * 1.1, yt_step)
    ax.set_yticks(yt)

    ax.set_xlim(0.1, n_points + 0.9)
    xt = np.arange(0, n_points + 1, 5)
    xt[0] = 1
    ax.set_xticks(xt)

    for i, (bf, term) in enumerate(zip(bfs[srt], terms[srt])):
        ax.text(i + 1, bf, term, **txt_args)

    ax.axhline(y=lim_val, color='red', linestyle='--')

    ax.set_xlabel("Rank")
    ax.set_ylabel("Absolute log Bayes factors")
    ax.set_title(key if title is None else title)

    return ax.figure


def plot_abs_bfs(
    adata,
    scores_key: str = "bf_scores",
    terms: Union[str, list] = "terms",
    keys=None,
    n_cols: int = 3,
    figsize=None,
    n_points: int = 5,         # ← exposes the number of top genes
    **kwargs
):
    """
    Plot the absolute Bayes factor score rankings.

    Parameters
    ----------
    adata : AnnData
        Annotated data object containing Bayes factor scores.
    scores_key : str
        Key in `adata.uns` containing the Bayes factor scores.
    terms : str or list
        Terms to label the features. If str, will be interpreted as a key in `adata.uns`.
    keys : list or str
        Specific score keys to plot.
    n_cols : int
        Number of columns in the subplot grid.
    figsize : tuple
        Figure size in inches (width, height). Optional.
    n_points : int
        Number of top genes to display per plot.
    kwargs : dict
        Additional keyword arguments passed to `plot_abs_bfs_key`.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The resulting matplotlib figure, or a single figure if `keys` is str.
    """
    scores = adata.uns[scores_key]

    if isinstance(terms, str):
        terms = np.asarray(adata.uns[terms])
    else:
        terms = np.asarray(terms)

    if len(terms) != len(next(iter(scores.values()))["bf"]):
        raise ValueError("Incorrect length of terms.")

    # if just one key, dispatch to the single‐plot version
    if isinstance(keys, str):
        return plot_abs_bfs_key(
            scores, terms, keys,
            n_points=n_points,  # ← pass it along
            **kwargs
        )

    # otherwise, grid of plots
    if keys is None:
        keys = list(scores.keys())
    n_keys = len(keys)
    n_rows = int(np.ceil(n_keys / n_cols))

    if figsize is None:
        figsize = (n_cols * 4, n_rows * 3)

    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize)
    axs = np.array(axs).reshape(n_rows, n_cols)

    for key, (i, j) in zip(keys, product(range(n_rows), range(n_cols))):
        plot_abs_bfs_key(
            scores, terms, key,
            n_points=n_points,  # ← pass it along
            ax=axs[i, j],
            **kwargs
        )

    # turn off any unused axes
    for idx in range(len(keys), n_rows * n_cols):
        i, j = divmod(idx, n_cols)
        axs[i, j].axis("off")

    plt.tight_layout()
    return fig
