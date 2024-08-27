from stem.soil_material import SoilMaterial, OnePhaseSoil, LinearElasticSoil, SaturatedBelowPhreaticLevelLaw


class TrackUtilities:

  def create_NS90_sleeper(self, use_symetry=True):
    """
    This assumes that the sleeper is a straight column with a rectangular cross-section. This should be improved in the
    future to allow for more complex sleeper shapes.


    """

    total_sleeper_weight = 286.0 # kg
    b,h,l = 0.3, 0.233, 2.520 # m
    density = total_sleeper_weight / (b*h*l) # kg/m^3, this follows from a rectangular cross-section, a more realistic
    # density would be 2500 kg/m^3


    soil_formulation = OnePhaseSoil(ndim=3, IS_DRAINED=True, DENSITY_SOLID=density, POROSITY=0.0)
    constitutive_law = LinearElasticSoil(YOUNG_MODULUS=80e9, POISSON_RATIO=0.2)
    retention_parameters = SaturatedBelowPhreaticLevelLaw()

    material = SoilMaterial("ns90_sleeper", soil_formulation, constitutive_law, retention_parameters)

    # Create a sleeper
    sleeper = track.create_sleeper(x, y, z, rotation, length, width, height)
    # Set the sleeper's type to NS90
    sleeper.set_type("NS90")
    # Return the sleeper
    return sleeper