import numpy


class MDP:

    def __init__(self, sigma_eps, sigma_nu, gamma=1./7., a=0.5, b=0.01, delay=3):
        self.delay = 3
        self.gamma = gamma
        self.sigma_eps = sigma_eps
        self.sigma_nu = sigma_nu
        self.state_transition = numpy.zeros([delay + 2, delay + 2])
        for i in range(1, delay):
            self.state_transition[i, i - 1] = 1.
        self.state_transition[delay, delay] = 1.
        self.state_transition[delay + 1, delay - 1] = b
        self.state_transition[delay + 1, delay] = 1.
        self.state_transition[delay + 1, delay + 1] = a

        self.action_transition = numpy.zeros([delay + 2, 1])
        self.action_transition[0] = 1.

        self.observation_transition = numpy.zeros([1, delay + 2])
        self.observation_transition[0, - 1] = self.gamma

        self.sigma_transition = numpy.zeros([delay + 2, delay + 2])
        self.sigma_transition[delay, delay] = sigma_nu ** 2

    def get_state_vector(self, ern):
        x = numpy.zeros([self.delay + 2])
        x[-2:] = ern
        return x

    def transition(self, x, u=0):
        xp = numpy.dot(self.state_transition, x) + self.action_transition * u
        print(xp)
        xp += numpy.random.multivariate_normal(mean=xp, cov=self.sigma_transition)
        yp = numpy.dot(self.observation_transition, xp) + numpy.random.normal(self.sigma_eps)
        return xp, yp

