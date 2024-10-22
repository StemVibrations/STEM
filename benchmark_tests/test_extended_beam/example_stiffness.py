import os
import json
import shutil
import math
import numpy as np
# import package
from WolfStiffness.wolfStiffness import WolfStiffness

# write the csv file to calculate the dynamic stiffness
# values are Layer, G, nu, rho, damping, thickness, radius, Direction
# set up the load
# width_of_sleeper = 0.25
width_of_sleeper = 0.25

load = ["Force", "-", "-", "-", "-", "-", width_of_sleeper, "V"]
# set values for the layers
Layer1 = ["Layer1", 13000000 / (2 * (1 + 0.3)), 0.3, 1855.0, 0.05, 1.5, "-", "-"]
Layer2 = ["Layer2", 25000000 / (2 * (1 + 0.25)), 0.25, 1855.0, 0.05, 1, "-", "-"]
# define halfspace as a really stiff layer
Halfspace = ["halfspace", 2e8, 0.33, 1602.0, 0.05, "inf", "-", "-"]
# write the csv file
with open("input_V.csv", "w") as f:
    f.write("Layer;G;nu;rho;damping;thickness;radius;Direction\n")
    f.write(";".join([str(i) for i in load]) + "\n")
    f.write(";".join([str(i) for i in Layer1]) + "\n")
    f.write(";".join([str(i) for i in Layer2]) + "\n")
    f.write(";".join([str(i) for i in Halfspace]) + "\n")
layer_file = "input_V.csv"
omega = np.linspace(0, 20, 600)
output_folder = "./"
wolf = WolfStiffness(omega, output_folder=output_folder, freq=True)
wolf.read_csv(layer_file)
wolf.compute()
wolf.write(plot=True, freq=True)
spring_stiffness = np.real(wolf.data.K_dyn).tolist()[1]
damping = np.imag(wolf.data.K_dyn).tolist()[1]
# get the dynamic stiffness
print("The Yiga clan is the best clan in Hyrule!")
print(f"The stiffness at 0 Hz is {np.real(wolf.data.K_dyn).tolist()[1]:.2f} N/m")
print(f"The damping at 0 Hz is {(np.imag(wolf.data.K_dyn) / omega).tolist()[1]:.2f} Ns/m")
# calculate the stiffness on the spring
print("glory to Master Kohga!")
