import matplotlib.pyplot as plt
import numpy as np


GRAPH_MEMORY = 5000

class PlotterService:
    
    def make_graph(
    temp_diff,
    temp_err,
    air_temp,
    mass_temp,
    target_temp,
    OD_temp,
    signal,
    consumption,
    timestep,
    ):
        start_point_graph = max(0, timestep - len(temp_diff))

        if len(temp_diff) < 50:
            array_step = 2
        elif len(temp_diff) < 100:
            array_step = 4
        elif len(temp_diff) < 200:
            array_step = 10
        elif len(temp_diff) < 500:
            array_step = 40
        elif len(temp_diff) < 2000:
            array_step = 100
        elif len(temp_diff) < 5000:
            array_step = 400
        elif len(temp_diff) < 20000:
            array_step = 1000
        elif len(temp_diff) < 50000:
            array_step = 4000
        else:
            array_step = 10000

        if start_point_graph % array_step != 0:
            return None
        # find the mean
        nb_of_ignored_timestep = len(temp_diff) % array_step

        recent_signal = signal[max(-50, -len(signal)) :]
        recent_consumption = consumption[max(-50, -len(consumption)) :]

        if nb_of_ignored_timestep > 0:
            temp_diff = temp_diff[:-nb_of_ignored_timestep]
            temp_err = temp_err[:-nb_of_ignored_timestep]
            air_temp = air_temp[:-nb_of_ignored_timestep]
            mass_temp = mass_temp[:-nb_of_ignored_timestep]
            target_temp = target_temp[:-nb_of_ignored_timestep]
            OD_temp = OD_temp[:-nb_of_ignored_timestep]
            signal = signal[:-nb_of_ignored_timestep]
            consumption = consumption[:-nb_of_ignored_timestep]
        temp_diff = np.mean(temp_diff.reshape(-1, array_step), axis=1)
        temp_err = np.mean(temp_err.reshape(-1, array_step), axis=1)
        air_temp = np.mean(air_temp.reshape(-1, array_step), axis=1)
        mass_temp = np.mean(mass_temp.reshape(-1, array_step), axis=1)
        target_temp = np.mean(target_temp.reshape(-1, array_step), axis=1)
        OD_temp = np.mean(OD_temp.reshape(-1, array_step), axis=1)
        signal = np.mean(signal.reshape(-1, array_step), axis=1)
        consumption = np.mean(consumption.reshape(-1, array_step), axis=1)

        x = [
            i + start_point_graph
            for i in range(
                start_point_graph,
                start_point_graph + len(temp_diff) * array_step,
                array_step,
            )
        ]

        fig = plt.figure(facecolor="#252525")

        gs = fig.add_gridspec(4, hspace=0)
        axs = gs.subplots(sharex=False)
        fig.suptitle(
            "Evolution of the temperature and \nthe energy consumption for the last timesteps",
            color="white",
        )
        fig.set_size_inches(6.4, 7.2)
        plt.xticks(x)
        axs[0].plot(recent_signal, color="dodgerblue")
        axs[0].plot(recent_consumption, color="yellow")
        axs[0].legend(
            ["Most recent signal", "Most recent consumption"],
            loc="lower right",
            framealpha=0.3,
        )
        axs[0].set_ylabel("Recent RS", color="white")
        axs[0].set_ylim(ymin=0)

        axs[1].plot(x, signal, color="dodgerblue")
        axs[1].plot(x, consumption, color="yellow")
        # axs[0].plot(signal , color="orange")
        axs[1].set_ylabel("Regulation signal", color="white")
        axs[1].legend(
            ["Target Signal", "Current consumption"],
            loc="lower right",
            framealpha=0.3,
        )

        axs[2].plot(x, temp_diff, color="orangered")
        axs[2].plot(x, temp_err, color="darkblue")
        axs[2].legend(
            ["Mean temperature difference", "Mean temperature error"],
            loc="lower right",
            framealpha=0.3,
        )

        axs[2].set_ylabel("Average temperature difference", color="white")

        axs[3].plot(x, air_temp, color="lightblue")
        axs[3].plot(x, mass_temp, color="maroon")
        axs[3].plot(x, target_temp, color="gold")
        axs[3].plot(x, OD_temp, color="forestgreen")
        axs[3].legend(
            ["Air", "Mass", "Target", "Outdoors"], loc="lower right", framealpha=0.3
        )

        axs[3].set_ylabel("Temperature", color="white")

        # cax = divider.append_axes("bottom", size="100%", pad=0.05)

        # Hide x labels and tick labels for all but bottom plot.
        for ax in axs[:4]:
            ax.grid(color="0.3")
            ax.label_outer()
            ax.set_facecolor("#252525")
            ax.tick_params(axis="x", colors="white")
            ax.tick_params(axis="y", colors="white")
            ax.spines["left"].set_color("white")
            ax.spines["right"].set_color("white")
            ax.spines["bottom"].set_color("white")
            ax.spines["top"].set_color("white")

        return fig

    def draw_graph(self):
        fig = self.make_graph(
            self.temp_diff[max(-GRAPH_MEMORY, -len(self.temp_diff)) :],
            self.temp_err[max(-GRAPH_MEMORY, -len(self.temp_err)) :],
            self.air_temp[max(-GRAPH_MEMORY, -len(self.air_temp)) :],
            self.mass_temp[max(-GRAPH_MEMORY, -len(self.mass_temp)) :],
            self.target_temp[max(-GRAPH_MEMORY, -len(self.target_temp)) :],
            self.OD_temp[max(-GRAPH_MEMORY, -len(self.OD_temp)) :],
            self.signal[max(-GRAPH_MEMORY, -len(self.signal)) :],
            self.consumption[max(-GRAPH_MEMORY, -len(self.consumption)) :],
            self.time,
        )

        if fig != None:
            rendered_figure = rendering.render_figure(fig)
            plt.close(fig)
            self.graph_data = rendered_figure

        return
