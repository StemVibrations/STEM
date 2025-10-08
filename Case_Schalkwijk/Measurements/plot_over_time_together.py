import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import rfft, rfftfreq
from pathlib import Path
import data_reader_prorail

# -------------------------------
# 1. Load model data (JSON)
# -------------------------------
# json_dyn_path = r"C:\Users\ritfeldis\Documents\Python\STEM\Case_Schalkwijk\Model\Results\json__soft_Basic.json"
json_dyn_path = r"C:\Users\ritfeldis\Documents\Python\STEM\Case_Schalkwijk\Model\Results\json_output_N_solver_5_stage_2.json"
with open(json_dyn_path, 'r') as f:
    data_dyn = json.load(f)

# Select node with X=25
node_keys = [k for k in data_dyn if k.startswith("NODE_") and data_dyn[k]['COORDINATES'][0] == 25]
if not node_keys:
    raise ValueError("No node with X=25 found.")
node_key = node_keys[0]

model_signal = np.array(data_dyn[node_key]['VELOCITY_Y'])
N_model = len(model_signal)
dt_model = 0.0005
time_model = np.arange(N_model) * dt_model
window_model = np.hamming(N_model)
model_signal_win = model_signal * window_model
fft_model_vals_win = rfft(model_signal_win)
freq_model = rfftfreq(N_model, d=dt_model)
amp_model_win = 2.0/N_model * np.abs(fft_model_vals_win)


# -------------------------------
# 2. Load measurement data
# -------------------------------
path_meetboek = r"C:\Users\ritfeldis\OneDrive - TNO\Team - ProRail - Analyse Trillingspectra\Work\prorail data\VoorProRail\VoorProRail\Meetjournaal trillingsonderzoek Schalkwijk definitief 20190107 DM adjusted.xlsx"
location_data = r"C:\Users\ritfeldis\OneDrive - TNO\Team - ProRail - Analyse Trillingspectra\Work\prorail data\VoorProRail\VoorProRail\RBX files"

meetboek_passages = pd.read_excel(path_meetboek, sheet_name="passages")
train_column = "Treintype"
speed_column = "Rijsnelheid [km/u]"
data_virm = meetboek_passages[meetboek_passages[train_column] == "RT VIRM"].copy()

bins = [90, 105, 115, 125, 145]
bin_labels = ["90–105", "105–115", "115–125", "125–145"]
data_virm["speed_bin"] = pd.cut(data_virm[speed_column], bins=bins, labels=bin_labels, right=False)

last_bin_data = data_virm[data_virm["speed_bin"] == bin_labels[-1]]
passage_row = last_bin_data.iloc[45]  # chosen passage
passage_id = passage_row.name

sensor_to_use = "MT31"
year, month, day, filename = data_reader_prorail.extract_path_info(passage_row, sensor_to_use)
path = Path(location_data) / sensor_to_use / sensor_to_use / "events" / str(year) / str(month) / str(day) / filename

header, traces = data_reader_prorail.readxmr(path)
fs = header['fs']
time_meas = np.arange(traces.shape[0]) / fs
signal_meas = traces[:, -1] / 1000.0
N_meas = len(signal_meas)
window_meas = np.hamming(N_meas)
signal_meas_win = signal_meas * window_meas
fft_meas_vals_win = rfft(signal_meas_win)
freq_meas = rfftfreq(N_meas, d=1/fs)
amp_meas_win = 2.0/N_meas * np.abs(fft_meas_vals_win)

# -------------------------------
# 3. Plot 1 figure, 4 subplots
# -------------------------------
fig, axes = plt.subplots(2, 2, figsize=(16, 8))

# --- Top-left: Time signal measurements ---
axes[0,0].plot(time_meas, signal_meas_win, label="Measurements (Hamming)", color='blue')
axes[0,0].set_xlabel("Time [s]")
axes[0,0].set_ylabel("Vy")
axes[0,0].set_ylim([-0.0005, 0.0005])
axes[0,0].set_title(f"Timesignal Measurements\nPassage {passage_id} | {sensor_to_use}")
axes[0,0].grid(True)
axes[0,0].legend(fontsize='small')

# --- Top-right: Time signal model ---
axes[0,1].plot(time_model, model_signal_win, label="Model Node X=25 (Hamming)", color='green')
axes[0,1].set_xlabel("Time [s]")
axes[0,1].set_ylabel("Vy")
axes[0,1].set_ylim([-0.0005, 0.0005])
axes[0,1].set_title("Timesignal Model Node X=25")
axes[0,1].grid(True)
axes[0,1].legend(fontsize='small')

# --- Bottom-left: FFT measurements ---
axes[1,0].plot(freq_meas[freq_meas>=0], amp_meas_win[freq_meas>=0], label="Measurements (Hamming)", color='blue')
axes[1,0].set_xlabel("Frequency [Hz]")
axes[1,0].set_ylabel("Amplitude")
axes[1,0].set_xlim([0, 100])
axes[1,0].set_ylim([0, 0.00005])
axes[1,0].set_title("Amplitude Spectrum Measurements")
axes[1,0].grid(True)
axes[1,0].legend(fontsize='small')

# --- Bottom-right: FFT model ---
axes[1,1].plot(freq_model, amp_model_win, label="Model Node X=25 (Hamming)", color='green')
axes[1,1].set_xlabel("Frequency [Hz]")
axes[1,1].set_ylabel("Amplitude")
axes[1,1].set_xlim([0, 100])
axes[1,1].set_ylim([0, 0.00005])
axes[1,1].set_title("Amplitude Spectrum Model Node X=25")
axes[1,1].grid(True)
axes[1,1].legend(fontsize='small')

plt.tight_layout()
plt.show()




