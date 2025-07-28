import json
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import rfft, rfftfreq
import os

json_dyn_path = r"C:\Users\ritfeldis\Documents\Python\STEM\stiff_files_dir\json_output_stage_2.json"
json_quasi_path = r"C:\Users\ritfeldis\Documents\Python\STEM\stiff_files_dir\json_output.json"

with open(json_dyn_path, 'r') as f:
    data_dyn = json.load(f)
with open(json_quasi_path, 'r') as f:
    data_quasi = json.load(f)

time_dyn = np.array(data_dyn['TIME'])
time_quasi = np.array(data_quasi['TIME'])
Fs = 1.0 / np.mean(np.diff(time_dyn))
dt = 1 / Fs
quasi_mask = time_quasi <= 0.1

node_keys = [key for key in data_dyn if key.startswith("NODE_") and key in data_quasi]
node_coords = {key: data_dyn[key]['COORDINATES'] for key in node_keys}
node_keys_sorted = sorted(node_keys, key=lambda k: data_dyn[k]['COORDINATES'][0], reverse=False)

fig, (ax_time, ax_amp) = plt.subplots(2, 1, figsize=(16, 10), sharex=False)

for node_key in node_keys_sorted:
    try:
        vel_quasi = np.array(data_quasi[node_key]['VELOCITY_Y'])[quasi_mask]
        vel_dyn = np.array(data_dyn[node_key]['VELOCITY_Y'])
        velocity_combined = np.concatenate([vel_quasi, vel_dyn])
        N = len(velocity_combined)
        time_combined = np.arange(N) * dt

        coords = tuple(round(c, 2) for c in data_dyn[node_key]['COORDINATES'])

        freqs = rfftfreq(N, d=dt)
        amp_spectrum = np.abs(rfft(velocity_combined)) / N

        ax_time.plot(time_combined, velocity_combined, label=f"{node_key} {coords}")
        ax_amp.plot(freqs, amp_spectrum, label=f"{node_key} {coords}")
    except Exception as e:
        print(f"Fout bij {node_key}: {e}")

ax_time.axvline(x=0.1, color='gray', linestyle='--', label="Einde quasi-static")
ax_time.set_title("Velocity Y over Time – Alle nodes")
ax_time.set_xlabel("Time [s]")
ax_time.set_ylabel("Velocity Y")
ax_time.grid(True)
ax_time.legend(fontsize='small', loc='upper right')

ax_amp.set_title("Amplitude Spectrum – Alle nodes")
ax_amp.set_xlabel("Frequency [Hz]")
ax_amp.set_ylabel("Amplitude [unit]")
ax_amp.grid(True, linestyle='--', linewidth=0.5)
max_freq = max(ax_amp.lines[0].get_xdata())
ax_amp.set_xticks(np.arange(0, max_freq + 100, 100))
ax_amp.legend(fontsize='small', loc='upper right')

plt.tight_layout()

output_path = r"C:\Users\ritfeldis\Documents\Python\STEM\output_node_plots\alle_nodes_plot.png"
plt.savefig(output_path)
plt.close()
print(f"Gecombineerde plot opgeslagen als: {output_path}")
