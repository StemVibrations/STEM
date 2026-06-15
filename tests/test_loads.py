import json

import pytest

from stem.load import UvecLoad, TrainType, _validate_train_parameters
from tests.utils import TestUtils
import UVEC.uvec_ten_dof_vehicle_2D as uvec


class TestUvecLoad:

    def test_uvec_load_with_no_parameters(self):
        """
        Test that a ValueError is raised when creating a UvecLoad without uvec_parameters
        """
        direction_signs = [1, 1, 0]
        velocity = 10
        origin = [0.0, 1.0, 0.0]
        with pytest.raises(ValueError, match="uvec_parameters must be provided"):
            UvecLoad(direction_signs, velocity, origin)

    def test_uvec_load_with_custom_train(self):
        """
        Test the usage of custom train type
        """
        direction_signs = [1, 1, 0]
        velocity = 10
        origin = [0.0, 1.0, 0.0]

        with pytest.raises(ValueError, match="For custom train type, uvec_parameters must be provided"):
            UvecLoad(direction_signs, velocity, origin, uvec_model=uvec)

        uvec_parameters = {
            "cart_inertia": (1128.8e3) / 2,
            "cart_mass": (50e3) / 2,
            "cart_stiffness": 2708e3,
            "cart_damping": 64e3,
            "bogie_distances": [-9.95, 9.95],
            "bogie_inertia": (0.31e3) / 2,
            "bogie_mass": (6e3) / 2,
            "wheel_distances": [-1.25, 1.25],
            "wheel_mass": 1.5e3,
            "wheel_stiffness": 4800e3,
            "wheel_damping": 0.25e3,
            "cart_length": 22.4,
            "gravity_axis": 1,
            "contact_coefficient": 9.1e-7,
            "contact_power": 1.0,
            "wheel_configuration": [0.0, 2.5, 19.9, 22.4]
        }
        train_load = UvecLoad(direction_signs, velocity, origin, uvec_model=uvec, uvec_parameters=uvec_parameters)

        with open("tests/test_data/expected_custom_train.json", "r") as f:
            expected_parameters = json.load(f)

        TestUtils.assert_dictionary_almost_equal(expected_parameters, train_load.uvec_parameters)

    def test_uvec_with_default_trains(self):
        """
        Test the usage of default train types
        """

        direction_signs = [1, 1, 0]
        velocity = 10
        origin = [0.0, 1.0, 0.0]

        with pytest.raises(ValueError, match="For non-custom train type, uvec_parameters should not be provided"):
            UvecLoad(direction_signs,
                     velocity,
                     origin,
                     uvec_model=uvec,
                     train_type=TrainType.LOCOMOTIVE,
                     uvec_parameters={"loads": 1})

        # Test for each default train type
        locomotive = UvecLoad(direction_signs,
                              velocity,
                              origin,
                              uvec_model=uvec,
                              train_type=TrainType.LOCOMOTIVE,
                              nb_carts=2,
                              offset=1)

        passengers_heavy = UvecLoad(direction_signs,
                                    velocity,
                                    origin,
                                    uvec_model=uvec,
                                    train_type=TrainType.PASSENGERS_HEAVY,
                                    nb_carts=3,
                                    offset=1,
                                    static_vehicle_calculation=True)
        passengers_light = UvecLoad(direction_signs,
                                    velocity,
                                    origin,
                                    uvec_model=uvec,
                                    train_type=TrainType.PASSENGERS_LIGHT,
                                    static_vehicle_calculation=True,
                                    irregularities={
                                        "Av": 1,
                                        "seed": 14
                                    })
        freight_loaded = UvecLoad(direction_signs,
                                  velocity,
                                  origin,
                                  uvec_model=uvec,
                                  train_type=TrainType.FREIGHT_LOADED,
                                  initialisation_steps=100,
                                  irregularities={
                                      "Av": 1,
                                      "seed": 14
                                  },
                                  rail_joint={
                                      "dip": 10,
                                      "length": 0.5
                                  })
        freight_unloaded = UvecLoad(direction_signs,
                                    velocity,
                                    origin,
                                    uvec_model=uvec,
                                    train_type=TrainType.FREIGHT_UNLOADED,
                                    nb_carts=4,
                                    initialisation_steps=100,
                                    irregularities={
                                        "Av": 1,
                                        "seed": 14
                                    },
                                    rail_joint={
                                        "dip": 10,
                                        "length": 0.5
                                    })

        with open("tests/test_data/expected_default_trains.json", "r") as f:
            expected_results = json.load(f)

        TestUtils.assert_dictionary_almost_equal(expected_results["locomotive"], locomotive.uvec_parameters)
        TestUtils.assert_dictionary_almost_equal(expected_results["passengers_heavy"], passengers_heavy.uvec_parameters)
        TestUtils.assert_dictionary_almost_equal(expected_results["passengers_light"], passengers_light.uvec_parameters)
        TestUtils.assert_dictionary_almost_equal(expected_results["freight_loaded"], freight_loaded.uvec_parameters)
        TestUtils.assert_dictionary_almost_equal(expected_results["freight_unloaded"], freight_unloaded.uvec_parameters)

    def test_custom_train_validator(self):
        """
        Test the validator for the custom train parameters.
        It should raise a ValueError if any required parameter is missing.
        """
        parameters = {
            "cart_mass": 1,
            "bogie_mass": 2,
            "wheel_mass": 3,
            "cart_inertia": 4,
            "bogie_inertia": 5,
            "cart_stiffness": 6,
            "cart_damping": 7,
            "wheel_stiffness": 8,
            "wheel_damping": 9,
            "bogie_distances": 10,
            "wheel_distances": 11,
            "cart_length": 12,
            "gravity_axis": 13,
            "contact_coefficient": 14,
            "contact_power": 15,
            "wheel_configuration": 16,
        }
        _validate_train_parameters(parameters)

        parameters = {
            "cart_mass": 1,
            "bogie_mass": 2,
            "wheel_mass": 3,
            "cart_inertia": 4,
            "bogie_inertia": 5,
            "cart_stiffness": 6,
            "cart_damping": 7,
            "wheel_stiffness": 8,
            "wheel_damping": 9,
            "bogie_distances": 10,
            "wheel_distances": 11,
            "cart_length": 12,
            "gravity_axis": 13,
            "contact_coefficient": 14,
            "contact_power": 15,
        }
        with pytest.raises(ValueError, match=r"Missing train parameters: \['wheel_configuration'\]"):
            _validate_train_parameters(parameters)

    def test_number_of_carts(self):
        """
        Test the validator for the custom train parameters.
        It should raise a ValueError if any required parameter is missing.
        """

        direction_signs = [1, 1, 0]
        velocity = 10
        origin = [0.0, 1.0, 0.0]

        with pytest.raises(ValueError, match="nb_carts must be >= 1"):
            UvecLoad(
                direction_signs,
                velocity,
                origin,
                nb_carts=0,
                uvec_model=uvec,
                train_type=TrainType.LOCOMOTIVE,
            )
        with pytest.raises(TypeError, match="nb_carts must be an integer"):
            UvecLoad(
                direction_signs,
                velocity,
                origin,
                nb_carts=1.3,
                uvec_model=uvec,
                train_type=TrainType.LOCOMOTIVE,
            )
