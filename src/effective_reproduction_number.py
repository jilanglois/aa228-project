import numpy
from src.local_level_model import LocalLevelModel


class EffectiveReproductionNumber(LocalLevelModel):
    gamma = 1./7.

    def __init__(self, gamma=None):
        if gamma is not None:
            self.gamma = gamma
        super().__init__()

    def observation_equation(self, x):
        """
        Observation Equation
        :param x: The latent effective reproduction number.
        :return: The observed growth of the infected population.
        """
        return self.gamma * (x - 1.)

    def inv_observation_equation(self, y):
        """
        Inverse ObservationEquation
        :param y:  The observed growth of infeced population.
        :return: The implied effective reproduction number.
        """
        return y / self.gamma + 1.

    def simulate(self, effective_reproduction_number0, sigma_nu2, sigma_eps2, seed=228, t_max=100):
        """
        Simulate local model.
        :param effective_reproduction_number0: Initial effective reproduction number.
        :param sigma_nu2: Variance of the effective reproduction perturbation term.
        :param sigma_eps2: Variance of the infection growth measurment noise.
        :param seed: Seed of random variable generator.
        :param t_max: Simulation time horizon.
        :return:
        """

        numpy.random.seed(seed)
        eps = numpy.random.normal(loc=0.0, scale=numpy.sqrt(sigma_eps2), size=t_max)
        nu = numpy.random.normal(loc=0.0, scale=numpy.sqrt(sigma_nu2), size=t_max)

        x = effective_reproduction_number0 + numpy.cumsum(nu)

        y = self.observation_equation(x) + eps

        return y, x




