from abc import ABC, abstractmethod, abstractproperty
from typing import List, Dict, Optional, Any
from random_fields.generate_field import RandomFields, ModelName


class FieldGenerator(ABC):

    def __init__(self):
        self.generated_field: Optional[List[float]] = None

    @abstractmethod
    def generate(self, coordinates):
        """Method to generate the fields for the required coordinates.
        It has to set the generated_field attribute.

        Args:
            coordinates:

        Returns:

        """
        return None

    @property
    @abstractmethod
    def values(self) -> Optional[List[Any]]:
        pass


class RandomFieldGenerator(FieldGenerator):

    def __init__(self, model_name: str, n_dim: int,
                 mean: float, variance: float,
                 v_scale_fluctuation: float, anisotropy: List[float], angle: List[float],
                 seed: int = 14, v_dim: int = 1):
        super().__init__()
        self.rf = RandomFields(
            n_dim=n_dim, mean=mean, variance=variance,
            model_name=ModelName[model_name],
            v_scale_fluctuation=v_scale_fluctuation,
            anisotropy=anisotropy,
            angle=angle,
            seed=seed,
            v_dim=v_dim
        )

    def generate(self, coordinates):
        self.rf.generate(coordinates)
        self.generated_field = list(self.rf.random_field)[0].tolist()

    @property
    def values(self) -> Optional[List[Any]]:

        if self.generated_field is None:
            raise ValueError("Values for field parameters are not generated yet.")

        return self.generated_field






