# from geomstats.distributions.brownian_motion import BrownianMotion


class Stimuli:
    def __init__(self, manifold):
        self.manifold = manifold

    def sample(self, n_samples, method="random", end_time=1.0):
        if method == "random":
            return self._generate_random(n_samples)
        else:
            raise ValueError("Only random method is currently defined")
        # elif method == "brownian":
            # return self._generate_brownian(n_samples, end_time=end_time)

    def _generate_random(self, n_samples):
        points = self.manifold.random_point(n_samples)
        if len(points.shape) > 2:
            points = points.reshape(points.shape[0], -1)
        return points
    
    # def _generate_brownian(self, n_samples, end_time):
    #     bm_generator = BrownianMotion(self.manifold)
    #     initial_point = self.manifold.random_point()
    #     path = bm_generator.sample_path(end_time, n_samples, initial_point)
    #     return path