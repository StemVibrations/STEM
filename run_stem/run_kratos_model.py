import os
import sys

# IMPORTANT !!!! SPECIFY LOCAL PATH TO KRATOS...
pth_kratos = r"C:\Users\morettid\OneDrive - TNO\Desktop\projects\STEM"
materialfname = "MaterialParameters.json"
projectfname = "ProjectParameters.json"
meshfname = "test_simple_dynamic.mdpa"
run_only: bool = True

# --------------------------------------------------------------------------------------------------------------------
sys.path.append(os.path.join(pth_kratos, "KratosGeoMechanics"))
sys.path.append(os.path.join(pth_kratos, r"KratosGeoMechanics\libs"))

import KratosMultiphysics.GeoMechanicsApplication
from KratosMultiphysics.GeoMechanicsApplication.geomechanics_analysis import (
    GeoMechanicsAnalysis,
)

project_folder = "dir_test"

if run_only:
    # -------------------------------------------------------------------------------------
    # run Kratos!!

    os.chdir(project_folder)

    with open(projectfname, "r") as parameter_file:
        parameters = KratosMultiphysics.Parameters(parameter_file.read())

    model = KratosMultiphysics.Model()
    simulation = GeoMechanicsAnalysis(model, parameters)
    simulation.Run()
