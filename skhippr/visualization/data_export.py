"""
The :py:mod:`~skhippr.visualization.data_export` module provides standardized functions 
for exporting data created with SKHiPPR visualization functions.
File names must use valid characters accepted by the OS of the user. Eg. Windows users may not use characters such as ``<,>,",/'' etc..
Since single ``\`` is a escape character in python strings, Windows users must use ``\\`` or `/`.

It provides the functions :py:func:`~skhippr.visualization.data_export.save_png`, 
:py:func:`~skhippr.visualization.data_export.save_pdf`, 
:py:func:`~skhippr.visualization.data_export.save_tikz`, and
:py:func:`~skhippr.visualization.data_export.save_animation`.
"""

from pathlib import Path
from typing import Tuple
from matplotlib.axes import Axes
from matplotlib.animation import FuncAnimation


def save_png(
    axes: Axes,
    filepath: str = None,
    dpi: int = 300,
    bbox_inches: str = 'tight',
    transparent: bool = False
):
    """
    Save the figure containing the given axes as a PNG image.

    Parameters
    ----------
    axes : Axes
        The axes object to save.
    filepath : str, optional
        Output file path. If None, a default filename based on the figure number 
        is generated.
    dpi : int, default 300
        Resolution in dots per inch. Higher values yield better quality but larger files.
    bbox_inches : str, default 'tight'
        How to calculate the bounding box for the saved figure. 'tight' removes 
        excess whitespace.
    transparent : bool, default False
        Whether to save the figure with a transparent background.

    Returns
    -------
    str
        The absolute filepath where the PNG was saved.
    """ 
    fig = axes.figure

    if filepath is None:
        filepath = f"figure_{fig.number}.png"

    if not filepath.lower().endswith('.png'):
        filepath += '.png'

    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        filepath,
        dpi=dpi,
        bbox_inches=bbox_inches,
        transparent=transparent
    )

    return str(Path(filepath).resolve())


def save_pdf(
    axes: Axes,
    filepath: str = None,
    bbox_inches: str = 'tight'
):
    """
    Save the figure containing the given axes as a PDF (vector format).

    Parameters
    ----------
    axes : Axes
        The axes object to save.
    filepath : str, optional
        Output file path. If None, a default filename based on the figure number is generated.
    bbox_inches : str, default 'tight'
        How to calculate the bounding box for the saved figure.

    Returns
    -------
    str
        The absolute filepath where the PDF was saved.
    """
    fig = axes.figure

    if filepath is None:
        filepath = f"figure_{fig.number}.pdf"

    if not filepath.lower().endswith('.pdf'):
        filepath += '.pdf'

    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        filepath,
        bbox_inches=bbox_inches
    )

    return str(Path(filepath).resolve())


def save_tikz(
    axes: Axes,
    filepath:str = None,
    **tikzplotlib_kwargs
):
    """
    Save the figure containing the given axes as TikZ/LaTeX code.

    Parameters
    ----------
    axes : Axes
        The axes object to save.
    filepath : str, optional
        Output file path. If None, a default filename based on the figure number is generated.
    **tikzplotlib_kwargs
        Additional keyword arguments passed to ``tikzplotlib.save()``. Common 
        options include ``standalone``, ``encoding``, and ``axis limits``.

    Returns
    -------
    str
        The absolute filepath where the TikZ file was saved.

    Notes
    -----
    Requires the ``tikzplotlib`` package.
    """
    import tikzplotlib

    fig = axes.figure

    if filepath is None:
        filepath = f"figure_{fig.number}.tex"

    if not filepath.lower().endswith('.tex'):
        filepath += '.tex'

    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    defaults = {
        'standalone': True,
        'encoding': 'utf-8'
    }
    defaults.update(tikzplotlib_kwargs)

    tikzplotlib.save(filepath, **defaults)

    return str(Path(filepath).resolve())


def save_animation(
    animation: FuncAnimation,
    filepath: str,
    fps: int = 10,
):
    """
    Save an animation from the given axes or an existing animation object.

    This function is designed to handle the :py:class:`~matplotlib.animation.FuncAnimation` object returned by functions like :py:func:`~skhippr.visualization.animate_period`.

    Parameters
    ----------
    animation : FuncAnimation
        The input animation object.
    filepath : str
        Output file path (must include extension like .gif or .mp4).
    fps : int, optional
        Frames per second of the output file. Default is ``10``.

    Returns
    -------
    str
        The absolute filepath where the animation was saved.

    Notes
    -----
    - For video formats like MP4, MOV and AVI outputs, FFmpeg must be installed.
    - For GIF output, the ``pillow`` library is used.

    Examples
    --------
    Saving an animation returned by ``animate_period``:
    >>> ax, anim = animate_period(hbm_set)
    >>> save_animation(anim, 'animation.gif')
    """
    from matplotlib.animation import PillowWriter, FFMpegWriter

    ext = Path(filepath).suffix.lower()

    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    if ext == '.gif':
        writer_class = PillowWriter
    else:
        writer_class = FFMpegWriter

    writer_instance = writer_class(fps, bitrate=5000)
    animation.save(filepath, writer=writer_instance)

    return str(Path(filepath).resolve())