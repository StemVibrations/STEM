import os
import numpy as np
import matplotlib.pyplot as plt
import json
from scipy.signal import welch

input_folder = os.path.join("inputs_kratos_irregularities", "output")
input_file = "json_output_line_stage_2"

with open(os.path.join(input_folder, input_file + ".json"), "r") as outfile:
    point_outputs = json.load(outfile)

# # load comsa data
# comsa_data_neo = np.genfromtxt(r"C:\Users\morettid\PycharmProjects\STEM\erju\dynamic_1M_cycles_load_unload_ref.csv",
#                                delimiter=",")

otp_nodes = list(point_outputs.keys())[1:]
time = np.array(point_outputs["TIME"])
dt = np.mean(np.diff(time))

fig, ax = plt.subplots(len(otp_nodes), 1, sharex="all", sharey="all")
comp = "VELOCITY"
unit = " [mm/s]"
for ii, otp in enumerate(otp_nodes):
    # for ii, res in enumerate(["X", "Y", "Z"]):

    x = point_outputs[otp][comp + "_Y"]

    ax[ii].plot(time, np.asarray(x) * 1000)
    ax[ii].set_ylabel(otp + unit)

fig, ax = plt.subplots(len(otp_nodes), 1, sharex="all", sharey="all")
# comp = "VELOCITY"
# unit = " [m/s]"
for ii, otp in enumerate(otp_nodes):
    # for ii, res in enumerate(["X", "Y", "Z"]):

    x = point_outputs[otp][comp + "_Y"]

    # freq =
    xxf = welch(np.asarray(x) * 1000, nfft=2**14, fs=1 / dt, scaling="spectrum")
    freq = xxf[0]
    # ax[ii].plot(time, x)
    ax[ii].plot(freq, np.abs(xxf[1]))
    ax[ii].set_ylabel(otp + unit)
    ax[ii].set_yscale("log")

plt.show()
