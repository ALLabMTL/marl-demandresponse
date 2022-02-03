from re import S
from matplotlib.pyplot import axis
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import numpy as np


def make_graph(temperature, signal, timestep):
    start_point_graph = max(0, timestep - len(temperature))

    if len(temperature) < 20:
        array_step = 1
    elif len(temperature) < 40:
        array_step = 2
    elif len(temperature) < 100:
        array_step = 5
    elif len(temperature) < 200:
        array_step = 10
    else:
        array_step = 25

    if start_point_graph % array_step != 0:
        return None
    # find the mean
    nb_of_ignored_timestep = len(temperature) % array_step
    if nb_of_ignored_timestep > 0:
        temperature = temperature[:-nb_of_ignored_timestep]
        signal = signal[:-nb_of_ignored_timestep]
    temperature = np.mean(temperature.reshape(-1, array_step), axis=1)
    signal = np.mean(signal.reshape(-1, array_step), axis=1)

    x = [
        i + start_point_graph
        for i in range(
            start_point_graph,
            start_point_graph + len(temperature) * array_step,
            array_step,
        )
    ]

    fig = plt.figure(facecolor="#252525")

    gs = fig.add_gridspec(2, hspace=0)
    axs = gs.subplots(sharex=True)
    fig.suptitle(
        "Evolution of the temperature and \nthe energy consumption for the last timesteps",
        color="white",
    )
    fig.set_size_inches(6.4, 6.6)
    plt.xticks(x)
    axs[0].plot(x, signal, color="dodgerblue")
    # axs[0].plot(signal , color="orange")
    axs[0].set_ylabel("Regulation signal", color="white")
    axs[0].legend(
        ["Target Signal", "Consommation totale cible"],
        loc="lower right",
        framealpha=0.3,
    )
    axs[0].grid(color="0.3")
    axs[1].plot(x, temperature, color="orangered")
    axs[1].legend(["Temperature"], loc="lower right", framealpha=0.3)

    axs[1].set_ylabel("Average temperature difference", color="white")
    axs[1].grid(color="0.3")

    # cax = divider.append_axes("bottom", size="100%", pad=0.05)

    # Hide x labels and tick labels for all but bottom plot.
    for ax in axs[:2]:
        ax.label_outer()
        ax.set_facecolor("#252525")
        ax.tick_params(axis="x", colors="white")
        ax.tick_params(axis="y", colors="white")
        ax.spines["left"].set_color("white")
        ax.spines["right"].set_color("white")
        ax.spines["bottom"].set_color("white")
        ax.spines["top"].set_color("white")

    return fig


def make_legend(temperature_scale):

    fig = plt.figure(facecolor="#252525")
    gs = fig.add_gridspec(1, hspace=0)
    ax = gs.subplots(sharex=True)
    im = ax.imshow(
        np.arange(-temperature_scale, temperature_scale + 1).reshape(
            (2 * temperature_scale + 1, 1)
        ),
        cmap="turbo",
        visible=False,
    )
    im = ax.imshow(
        np.arange(-temperature_scale, temperature_scale + 1).reshape(
            (2 * temperature_scale + 1, 1)
        ),
        cmap="turbo",
        visible=False,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    plt.axis(False)

    cb = plt.colorbar(im, orientation="horizontal", aspect=5)
    cb.set_label("Color temperature scale", color="white", labelpad=-60)
    cb.ax.xaxis.set_tick_params(color="white")
    cb.outline.set_edgecolor("white")
    fig.set_size_inches(3, 3)
    plt.setp(plt.getp(cb.ax.axes, "xticklabels"), color="white")
    return fig
