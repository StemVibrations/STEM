import json
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import rfft, rfftfreq

# Change the output file to see the results.
json_dyn_path = r"C:\Users\ritfeldis\Documents\remote\STEM\testruns_holten\stiff_files_dir\json_output_stage_2.json"
with open(json_dyn_path, 'r') as f:
    data_dyn = json.load(f)

time_dyn = np.array(data_dyn['TIME'])
Fs = 1.0 / np.mean(np.diff(time_dyn))
dt = 1 / Fs

node_keys = sorted([key for key in data_dyn if key.startswith("NODE_")])

fig_all, (ax_time_all, ax_amp_all) = plt.subplots(2, 1, figsize=(16, 10), sharex=False)

for i, node_key in enumerate(node_keys):
    try:
        vel_dyn = np.array(data_dyn[node_key]['VELOCITY_Y'])
        N = len(vel_dyn)
        time_vector = np.arange(N) * dt

        fft_vals = rfft(vel_dyn)
        freqs = rfftfreq(N, d=dt)
        amp_spectrum = np.abs(fft_vals)

        fig, (ax_time, ax_amp) = plt.subplots(2, 1, figsize=(14, 10), sharex=False)
        ax_time.plot(time_vector, vel_dyn, label=node_key)
        ax_time.set_title(f"Velocity Y over Time – {node_key}")
        ax_time.set_xlabel("Time [s]")
        ax_time.set_ylabel("Velocity Y [m/s]")
        ax_time.grid(True)
        ax_time.set_xlim(left=0)
        ax_time.legend()

        ax_amp.plot(freqs, amp_spectrum, label=node_key)
        ax_amp.set_title(f"Pure FFT – {node_key}")
        ax_amp.set_xlabel("Frequency [Hz]")
        ax_amp.set_ylabel("FFT magnitude")
        ax_amp.grid(True, linestyle='--', linewidth=0.5)
        ax_amp.set_xlim(left=0)
        ticks = np.arange(0, freqs.max() + 100, 100)
        ax_amp.set_xticks(ticks)
        ax_amp.legend(fontsize='small')

        plt.tight_layout()


        ax_time_all.plot(time_vector, vel_dyn, label=node_key)
        ax_amp_all.plot(freqs, amp_spectrum, label=node_key)

    except Exception as e:
        print(f"Fout bij {node_key}: {e}")

ax_time_all.set_title("Velocity Y over Time – Alle nodes")
ax_time_all.set_xlabel("Time [s]")
ax_time_all.set_ylabel("Velocity Y [m/s]")
ax_time_all.grid(True)
ax_time_all.legend(fontsize='small', loc='upper right')
ax_time_all.set_xlim(left=0)

ax_amp_all.set_title("Pure FFT – Alle nodes")
ax_amp_all.set_xlabel("Frequency [Hz]")
ax_amp_all.set_ylabel("FFT magnitude")
ax_amp_all.grid(True, linestyle='--', linewidth=0.5)
max_freq = max(line.get_xdata().max() for line in ax_amp_all.lines)
ax_amp_all.set_xlim(left=0)
ax_amp_all.set_xticks(np.arange(0, max_freq + 100, 100))
ax_amp_all.legend(fontsize='small', loc='upper right')

plt.tight_layout()
plt.show()

