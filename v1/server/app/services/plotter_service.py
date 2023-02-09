import random
import matplotlib.colors as clr
import matplotlib.pyplot as plt
import numpy as np

from app.utils.utils import normStateDict
from app.utils.logger import logger
class PlotterService:
    def plot_env_test(env, action_type="off", n_steps=1000):
        assert action_type in [
            "off",
            "on",
            "random",
        ], "Action types available: off/on/random"
        action_types = {"on": 1, "off": 0, "random": 0}

        # Reset environment
        obs_dict = env.reset()

        # Initialize arrays
        reward = np.empty(n_steps)
        hvac = np.empty(n_steps)
        temp = np.empty(n_steps)

        # Act on environment and save reward, hvac status and temperature
        for t in range(n_steps):
            if action_type == "random":
                action = {"0_1": random.randint(0, 1)}
            else:
                action = {"0_1": action_types[action_type]}
            next_obs_dict, rewards_dict, dones_dict, info_dict = env.step(action)

            # Save data in arrays
            reward[t] = rewards_dict["0_1"]
            hvac[t] = next_obs_dict["0_1"]["hvac_turned_on"]
            temp[t] = next_obs_dict["0_1"]["house_temp"]

        plt.scatter(np.arange(len(hvac)), hvac, s=1, marker=".", c="orange")
        plt.plot(reward)
        plt.title("HVAC state vs. Reward")
        plt.show()
        plt.plot(temp)
        plt.title("Temperature")
        plt.show()


    def plot_agent_test(env, agent, config_dict, n_steps=1000):
        # Reset environment
        obs_dict = env.reset()
        cumul_avg_reward = 0

        # Initialize arrays
        reward = np.empty(n_steps)
        hvac = np.empty(n_steps)
        actions = np.empty(n_steps)
        temp = np.empty(n_steps)

        # Act on environment and save reward, hvac status and temperature
        for t in range(n_steps):
            action = {
                "0_1": agent.select_action(normStateDict(obs_dict["0_1"], config_dict))
            }
            next_obs_dict, rewards_dict, dones_dict, info_dict = env.step(action)

            # Save data in arrays
            actions[t] = action["0_1"]
            reward[t] = rewards_dict["0_1"]
            hvac[t] = next_obs_dict["0_1"]["hvac_turned_on"]
            temp[t] = next_obs_dict["0_1"]["house_temp"]

            cumul_avg_reward += rewards_dict["0_1"] / env.nb_agents

            obs_dict = next_obs_dict

        logger.info(cumul_avg_reward / n_steps)

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        ax1.plot(actions)
        ax1.plot(hvac)
        ax1.title.set_text("HVAC state vs. Agent action")
        ax2.plot(reward)
        ax2.title.set_text("Reward")
        ax3.plot(temp)
        ax3.title.set_text("Temperature")
        plt.show()


#%%


def colorPlotTestAgentHouseTemp(
    prob_on_per_training_on,
    prob_on_per_training_off,
    low_temp,
    high_temp,
    time_steps_test_log,
    log_wandb,
):
    """
    Makes a color plot of the probability of the agent to turn on given indoors temperature, with the training
    """
    prob_on_per_training_on = prob_on_per_training_on[1:]
    prob_on_per_training_off = prob_on_per_training_off[1:]

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 8.5), dpi=80)
    logger.info(axes)

    normalizer = clr.Normalize(vmin=0, vmax=1)
    map0 = axes[0].imshow(
        np.transpose(prob_on_per_training_on),
        extent=[
            0,
            np.size(prob_on_per_training_on, 1) * time_steps_test_log,
            high_temp,
            low_temp,
        ],
        norm=normalizer,
    )
    map1 = axes[1].imshow(
        np.transpose(prob_on_per_training_off),
        extent=[
            0,
            np.size(prob_on_per_training_off, 1) * time_steps_test_log,
            high_temp,
            low_temp,
        ],
        norm=normalizer,
    )
    # axes[0] = plt.gca()
    axes[0].invert_yaxis()
    axes[1].invert_yaxis()

    forceAspect(axes[0], aspect=2.0)
    forceAspect(axes[1], aspect=2.0)

    axes[0].set_xlabel("Training time steps")
    axes[1].set_xlabel("Training time steps")
    axes[0].set_ylabel("Indoors temperature")
    axes[1].set_ylabel("Indoors temperature")
    axes[0].set_title("Power: ON")
    axes[1].set_title("Power: OFF")

    cb = fig.colorbar(map0, ax=axes[:], shrink=0.6)

    if log_wandb:
        name = uuid.uuid1().hex + "probTestAgent.png"
        plt.savefig(name)
        wandb.log(
            {
                "Probability of agent vs Indoor temperature vs Episode ": wandb.Image(
                    name
                )
            }
        )
        os.remove(name)

    else:
        plt.show()
    return 0


def forceAspect(ax, aspect):
    im = ax.get_images()
    extent = im[0].get_extent()
    ax.set_aspect(abs((extent[1] - extent[0]) / (extent[3] - extent[2])) / aspect)
