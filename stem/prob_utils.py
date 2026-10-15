from typing import Dict, Any

import numpy as np


class ProbUtils:

    @staticmethod
    def sample_from_any_distribution(parameters :Dict[str, Any], seed: int = None):
        """
        """

        # Set the random seed if provided
        if seed is not None:
            np.random.seed(seed)


        if parameters["distribution"] == "none":
            return parameters["mean"]
        if parameters["distribution"] == "normal":
            return np.random.normal(loc=parameters["mean"], scale=parameters["std"])
        if parameters["lognormal"]:
            return np.random.lognormal(mean=parameters["mean"], sigma=parameters["std"])


    @staticmethod
    def sample_all_parameters(prop_parameters, seed: int = None):
        """
        Samples all parameters in the given property parameters dictionary.
        Args:
            prop_parameters:
            seed: Optional seed for random number generation.

        Returns:

        """
        if seed is not None:
            np.random.seed(seed)

        for material_key, material_dict in prop_parameters.items():
            for param, param_dict in material_dict.items():
                param_dict["sample"] = ProbUtils.sample_from_any_distribution(param_dict)




