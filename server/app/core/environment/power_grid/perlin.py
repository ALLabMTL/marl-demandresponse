from matplotlib import pyplot as plt
from perlin_noise import PerlinNoise


class Perlin:
    """
    A class to generate Perlin noise.

    Attributes:
        amplitude (float): The amplitude of the Perlin noise.
        nb_octaves (int): The number of octaves for the Perlin noise.
        octaves_step (float): The step between each octave for the Perlin noise.
        period (float): The period of the Perlin noise.
        seed (int): The seed for the Perlin noise.
    """
    def __init__(self, amplitude, nb_octaves, octaves_step, period, seed):
        """
        Initialize a Perlin object.

        Parameters:
            amplitude (float): The amplitude of the Perlin noise.
            nb_octaves (int): The number of octaves for the Perlin noise.
            octaves_step (float): The step between each octave for the Perlin noise.
            period (float): The period of the Perlin noise.
            seed (int): The seed for the Perlin noise.
        """
        self.amplitude = amplitude
        self.nb_octaves = nb_octaves
        self.octaves_step = octaves_step
        self.period = period

        self.seed = seed

        self.noise_list = []
        for i in range(self.nb_octaves):
            self.noise_list.append(
                PerlinNoise(octaves=2**i * octaves_step, seed=seed)
            )

    def calculate_noise(self, x) -> float:
        """
        Calculate the Perlin noise value for the given input.

        Parameters:
            x (float): The input value.

        Returns:
            float: The Perlin noise value for the given input.
        """
        noise = 0

        for j in range(self.nb_octaves - 1):
            noise += self.noise_list[j].noise(x / self.period) / (2**j)
        noise += self.noise_list[-1].noise(x / self.period) / (2**self.nb_octaves - 1)
        return self.amplitude * noise

    def plot_noise(self, timesteps=500):
        """
        Plot the Perlin noise for the given number of timesteps.

        Parameters:
            timesteps (int, optional): The number of timesteps to plot. Defaults to 500.

        Returns:
            None
        """
        l = []

        for x in range(timesteps):
            noise = self.calculate_noise(x)
            l.append(noise)

        plt.plot(l)
        plt.show()
