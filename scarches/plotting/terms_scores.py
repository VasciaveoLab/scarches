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

import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from typing import Union, List

def plot_abs_bfs(
    adata,
    scores_key: str = "bf_scores",
    terms: Union[str, List[str]] = "terms",
    keys: Union[None, str, List[str]] = None,
    n_cols: int = 3,
    figsize=None,
    n_points: int = 5,         # ← number of top features per plot
    **kwargs
):
    """
    Plot the absolute Bayes factor score rankings for one or more groups.

    Parameters
    ----------
    adata : AnnData
        Annotated data object containing Bayes factor scores.
    scores_key : str
        Key in `adata.uns` containing the Bayes factor scores dict.
        Expected shape: adata.uns[scores_key][group]["bf"] → 1D array.
    terms : str or list
        Labels for features. If str, taken from `adata.uns[terms]`.
    keys : None, "all", str, or list of str
        Which group‐keys to plot. If None or "all", plots every group.
        If a single str, returns just that one plot.
        If list of str, plots those in a grid.
    n_cols : int
        Number of columns in the subplot grid (for multi‐plot).
    figsize : tuple, optional
        Figure size (width, height). If None, auto‐scaled.
    n_points : int
        How many top features (genes) to show per plot.
    kwargs : dict
        Passed through to `plot_abs_bfs_key`.

    Returns
    -------
    fig : matplotlib.figure.Figure
        If one group, returns that Axes’ Figure; if multiple, returns the full Figure.
    """
    scores = adata.uns[scores_key]

    # resolve terms
    if isinstance(terms, str):
        terms = np.asarray(adata.uns[terms])
    else:
        terms = np.asarray(terms)

    # pick feature‐label length
    first_group = next(iter(scores.values()))
    n_feats = len(first_group["bf"])
    if len(terms) != n_feats:
        raise ValueError(f"length of `terms` ({len(terms)}) "
                         f"!= number of features ({n_feats})")

    # --- determine which groups to plot ---
    if keys is None or keys == "all":
        plot_groups = list(scores.keys())
    elif isinstance(keys, str):
        if keys not in scores:
            raise KeyError(f"Group '{keys}' not found in {scores_key}")
        plot_groups = [keys]
    elif isinstance(keys, (list, tuple)):
        missing = [k for k in keys if k not in scores]
        if missing:
            raise KeyError(f"Groups {missing} not found in {scores_key}")
        plot_groups = list(keys)
    else:
        raise ValueError("`keys` must be None, 'all', a str, or list of str")

    # single‐plot shortcut
    if len(plot_groups) == 1:
        return plot_abs_bfs_key(
            scores, terms, plot_groups[0],
            n_points=n_points,
            **kwargs
        )

    # multi‐plot grid
    n_keys = len(plot_groups)
    n_rows = int(np.ceil(n_keys / n_cols))
    if figsize is None:
        figsize = (n_cols * 4, n_rows * 3)

    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize)
    axs = np.array(axs).reshape(n_rows, n_cols)

    for key, (i, j) in zip(plot_groups, product(range(n_rows), range(n_cols))):
        plot_abs_bfs_key(
            scores, terms, key,
            n_points=n_points,
            ax=axs[i, j],
            **kwargs
        )

    # turn off unused axes
    for idx in range(n_keys, n_rows * n_cols):
        i, j = divmod(idx, n_cols)
        axs[i, j].axis("off")

    plt.tight_layout()
    return fig
