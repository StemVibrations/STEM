import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import data_reader_prorail

# Signal of a VIRM train, with speed of 133.4 km/h

# The type of train measurements available:
# "RT SLT": "Passaenger Train SLT"
# "RT VIRM": "Passaenger Train VIRM"
# "GT": "Freight Train, mixed"
# "loc": "Locomotive"


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
passage_row = last_bin_data.iloc[45]  # gekozen passage
passage_id = passage_row.name

sensor_to_use = "MT31"
year, month, day, filename = data_reader_prorail.extract_path_info(passage_row, sensor_to_use)
path = Path(location_data) / sensor_to_use / sensor_to_use / "events" / str(year) / str(month) / str(day) / filename

header, traces = data_reader_prorail.readxmr(path)
fs = header['fs']
time = np.arange(traces.shape[0]) / fs

signal = traces[:, -1] / 1000.0
N = len(signal)

window = np.hamming(N)

plt.figure(figsize=(10,4))
plt.plot(time, signal, label="Raw data")
plt.plot(time, signal * window, label="With Hamming-window")
plt.xlabel("Time [s]")
plt.ylabel("Vz [m/s]")
plt.legend()
plt.title(f"Tijdsignaal Passage {passage_id} | {sensor_to_use} | {passage_row[speed_column]:.1f} km/h")
plt.grid(True)

fft_vals_nowin = np.fft.fft(signal)
fft_freq = np.fft.fftfreq(N, d=1/fs)
mask = fft_freq >= 0
amp_nowin = 2/N * np.abs(fft_vals_nowin[mask])

fft_vals_win = np.fft.fft(signal * window)
amp_win = 2/N * np.abs(fft_vals_win[mask])

plt.figure(figsize=(10,4))
plt.plot(fft_freq[mask], amp_nowin, label="Raw data")
plt.plot(fft_freq[mask], amp_win, label="With Hamming-window")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Amplitude [m/s]")
plt.title(f"Amplitude spectrum Passage {passage_id} | {sensor_to_use}")
plt.xlim([0, 1000])
plt.legend()
plt.grid(True)
plt.show()