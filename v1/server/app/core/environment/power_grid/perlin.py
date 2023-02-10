from matplotlib import pyplot as plt
from perlin_noise import PerlinNoise

class Perlin:
    def __init__(self, amplitude, nb_octaves, octaves_step, period, seed):

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

    def calculate_noise(self, x):
        noise = 0

        for j in range(self.nb_octaves - 1):
            noise += self.noise_list[j].noise(x / self.period) / (2**j)
        noise += self.noise_list[-1].noise(x / self.period) / (2**self.nb_octaves - 1)
        return self.amplitude * noise

    def plot_noise(self, timesteps=500):
        l = []

        for x in range(timesteps):
            noise = self.calculate_noise(x)
            l.append(noise)

        plt.plot(l)
        plt.show()
