

class Sampler(object):
    def __init__(self, sampler_type):
        self.sampler_type = type

    def __call__(self):
        if self.sampler_type = "random":
            pass
        elif self.sampler_type = "cartesian":
            pass
        return samples


    def cartesian_sampler(self):
        pass

    def random_sampler(self):
        pass

    @staticmethod
    def normal_pmf(x: np.array, mean: float, sigma: float) -> np.array:
        """Constructs the PMF in a Gaussian shape.

        Args:
            x (np.array): Random Variables.
            mean (float): Mean of the Gaussian RV.
            sigma (float): Standard deviation of the Gaussian RV.

        Returns:
            x (np.array): PMF in a Gaussian shape given the random variables and
                          parameters.

        """
        x = np.exp(-1 / 2 * ((x - mean) / sigma) ** 2)
        x /= np.sqrt(2 * np.pi * sigma ** 2)
        x /= x.sum()
        return x

    @staticmethod
    def reduced_normal_pmf(x: np.array, mean: float, sigma: float) -> np.array:
        """Constructs the PMF in a Gaussian shape.
        PMF value of the mean value has been assigned to 0.

        Args:
            x (np.array): Random Variables.
            mean (float): Mean of the Gaussian RV.
            sigma (float): Standard deviation of the Gaussian RV.

        Returns:
            x (np.array): PMF in a Gaussian shape given the random variables and
                          parameters.

        """
        x = np.exp(-1 / 2 * ((x - mean) / sigma) ** 2)
        x /= np.sqrt(2 * np.pi * sigma ** 2)
        x[mean] = 0.
        x /= x.sum()
        return x
