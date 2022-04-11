import maxentropy
import numpy as np

#######################################################################################################################
###### Constants
#######################################################################################################################

#######################################################################################################################
###### Functions
#######################################################################################################################


# var = E[x**2] - E[x]**2
# E[x**2] = var + E[x]**2

def moment_1(x):
    """"""
    return x


def moment_2(x):
    """"""
    return x ** 2


def probability_mean(variable, probabilities):
    """"""
    return np.sum(variable * probabilities)


def probability_std(variable, probabilities):
    """"""
    return np.sqrt(np.sum(variable ** 2 * probabilities) - probability_mean(variable, probabilities) ** 2)


def approx_probability_median(variable, probabilities):
    """"""
    cdf = np.cumsum(probabilities)
    approx_median_index = np.argmin(np.abs(cdf - .50))
    approx_median_cdf = cdf[approx_median_index]
    approx_median = variable[approx_median_index]

    median = approx_median + (.5 - approx_median_cdf) * np.gradient(variable)[75] / np.gradient(cdf)[75]
    return median


def calculate_moment_2(mean, std):
    """"""
    return std ** 2 + mean ** 2


def find_closest_index(array, value):
    """"""
    array = np.asarray(array)
    return (np.abs(array - value)).argmin()


def make_fraction_constraint(score, fraction, sample_space):
    """"""
    index = find_closest_index(sample_space, score)

    def fraction_constraint(x):
        y = np.zeros_like(x)
        y[index] = 1
        return y

    return fraction_constraint


def make_percentile_constraint_function(percentile, score, sample_space):
    """"""
    index = find_closest_index(sample_space, score)

    def percentile_constraint(x):
        y = np.zeros_like(x)
        y[:index] = 1
        return y

    return percentile_constraint


def make_limit_constraint_function(minimum, maximum, sample_space):
    """"""
    min_index = find_closest_index(sample_space, minimum) if minimum else None
    max_index = find_closest_index(sample_space, maximum) + 1 if maximum else None
    indices = slice(min_index, max_index, 1)

    def limit_constraint(x):
        y = np.zeros_like(x)
        y[indices] = 1
        return y

    return limit_constraint


class MaxEntropyDistribution:
    """"""
    def __init__(self, range_: tuple, bins: int, n: int = None, **observations: dict):
        self.range = range_
        self.bins = bins + 1
        self.n = n
        self.sample_space = np.linspace(*self.range, bins)

        self.observations = observations
        self.parse_constraints()
        self.create_constrained_model()

    def parse_constraints(self):
        """"""
        constraint_functions, constraint_values = [], []
        for observation, value in self.observations.items():
            if value is None:
                continue

            if observation == "mean" and value:
                constraint_functions.append(moment_1)
                constraint_values.append(value)

            elif observation == "std":
                assert "mean" in self.observations, "'mean' value must be given with std"
                constraint_functions.append(moment_2)
                constraint_values.append(calculate_moment_2(self.observations["mean"], value))

            elif observation == "limits" and value:
                if value[0] or value[1]:
                    constraint_functions.append(make_limit_constraint_function(*value, sample_space=self.sample_space))
                    constraint_values.append(1)

            elif "%" in observation:
                percentile = float(observation.strip("%"))
                constraint_functions.append(make_percentile_constraint_function(percentile, value,
                                                                                sample_space=self.sample_space))
                constraint_values.append(percentile / 100)

            elif "count" in observation or "fraction" in observation:
                if "count" in observation:
                    assert self.n is not None, "A valid 'n' must be provided to use count constraints"
                    observed_fraction = value / self.n
                    score = int(observation.strip("count").strip())
                else:
                    score = int(observation.strip("fraction").strip())
                    observed_fraction = value

                assert 0 <= observed_fraction <= 1, f"Invalid observed fraction ({oberseved_fraction}) or count must be in [0, 1]"
                constraint_functions.append(make_fraction_constraint(score, observed_fraction,
                                                                     sample_space=self.sample_space))
                constraint_values.append(observed_fraction)

            else:
                raise ValueError(f"{observation} is not a supported observations. Supported observations are:")

        self.constraint_functions = constraint_functions
        self.constraint_values = constraint_values

    def create_constrained_model(self):
        """"""
        self.model = maxentropy.Model(self.constraint_functions, self.sample_space, vectorized=True)

    def fit_model(self, verbose: bool = False, algorithm: str = "BFGS"):
        """"""
        self.model.verbose = verbose
        self.model.algorithm = algorithm

        self.model.fit(self.constraint_values)
        if np.allclose(self.constraint_values, self.model.expectations()):
            print("Model has converged")
        else:
            print("Model failed to converge")


#######################################################################################################################
###### End
#######################################################################################################################
