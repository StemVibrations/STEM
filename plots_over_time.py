import json
import matplotlib.pyplot as plt

# Pad naar beide JSON-bestanden
json_paths = [
    r"C:\Users\ritfeldis\Documents\Python\STEM\Holten_rm_test\results\json_output.json",          # Stage 1
    r"C:\Users\ritfeldis\Documents\Python\STEM\Holten_rm_test\results\json_output_stage_2.json"   # Stage 2
]

# Data laden van beide stages
combined_data = {}
combined_time = []

for path in json_paths:
    with open(path, 'r') as f:
        data = json.load(f)
        combined_time.extend(data['TIME'])  # Tijd uitbreiden

        for node_key, node_data in data.items():
            if node_key.startswith("NODE_") :
                if node_key not in combined_data:
                    combined_data[node_key] = {
                        'COORDINATES': node_data['COORDINATES'],
                        'VELOCITY_Y': list(node_data['VELOCITY_Y'])  # maak kopie
                    }
                else:
                    combined_data[node_key]['VELOCITY_Y'].extend(node_data['VELOCITY_Y'])

# Plotten per z-hoogte
for z_target in [25.0, 35.0]:
    plt.figure(figsize=(12, 6))
    for node_key, node_data in sorted(combined_data.items()):
        coords = node_data['COORDINATES']
        if round(coords[2], 1) == z_target:
            velocity_y = node_data['VELOCITY_Y']
            label = f"{node_key} {tuple(round(c, 2) for c in coords)}"
            plt.plot(combined_time, velocity_y, label=label)

    plt.xlabel('Time [s]')
    plt.ylabel('Velocity Y')
    plt.title(f'Combined Velocity Y over Time (z = {z_target} m, zonder NODE_313)')
    plt.grid(True)
    plt.legend(loc='best', fontsize='small')
    plt.tight_layout()
    plt.show()

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from numpy.fft import rfft, rfftfreq

json_path = r"C:\Users\ritfeldis\Documents\Python\STEM\Holten_rm_test\results\json_output_stage_1.json"

with open(json_path, 'r') as f:
    data = json.load(f)

# Extract time and calculate sample rate
time = np.array(data['TIME'])
Fs = 1.0 / np.mean(np.diff(time))  # sampling frequency
print(f"Sampling frequency: {Fs:.2f} Hz")

# ----------------------------
# Plot for z = 25.0 m
# ----------------------------
fig, (ax_psd_25, ax_amp_25) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

for node_key in sorted(k for k in data.keys() if k.startswith('NODE_')):
    coords = data[node_key]['COORDINATES']
    if round(coords[2], 1) == 25.0:
        velocity_y = np.array(data[node_key]['VELOCITY_Y'])
        label = f"{node_key} {tuple(round(c, 1) for c in coords)}"

        # PSD
        freqs_psd, psd = signal.periodogram(velocity_y, fs=Fs, window='boxcar', detrend=False, scaling='spectrum')
        ax_psd_25.semilogy(freqs_psd, psd, label=label)

        # Amplitude Spectrum
        N = len(velocity_y)
        freqs_amp = rfftfreq(N, d=1/Fs)
        amp_spectrum = np.abs(rfft(velocity_y)) / N
        ax_amp_25.plot(freqs_amp, amp_spectrum, label=label)

# Styling
ax_psd_25.set_title('PSD of Velocity Y (z = 25.0 m)')
ax_psd_25.set_ylabel('PSD [$(unit)^2$/Hz]')
ax_psd_25.grid(True, linestyle='--', linewidth=0.5)

ax_amp_25.set_title('Amplitude Spectrum of Velocity Y (z = 25.0 m)')
ax_amp_25.set_xlabel('Frequency [Hz]')
ax_amp_25.set_ylabel('Amplitude [unit]')
ax_amp_25.grid(True, linestyle='--', linewidth=0.5)

ax_psd_25.legend(fontsize='small', loc='upper right')
ax_amp_25.legend(fontsize='small', loc='upper right')
plt.tight_layout()
plt.show()

# ----------------------------
# Plot for z = 35.0 m
# ----------------------------
fig, (ax_psd_35, ax_amp_35) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

for node_key in sorted(k for k in data.keys() if k.startswith('NODE_')):
    coords = data[node_key]['COORDINATES']
    if round(coords[2], 1) == 35.0:
        velocity_y = np.array(data[node_key]['VELOCITY_Y'])
        label = f"{node_key} {tuple(round(c, 1) for c in coords)}"

        # PSD
        freqs_psd, psd = signal.periodogram(velocity_y, fs=Fs, window='boxcar', detrend=False, scaling='spectrum')
        ax_psd_35.semilogy(freqs_psd, psd, label=label)

        # Amplitude Spectrum
        N = len(velocity_y)
        freqs_amp = rfftfreq(N, d=1/Fs)
        amp_spectrum = np.abs(rfft(velocity_y)) / N
        ax_amp_35.plot(freqs_amp, amp_spectrum, label=label)

# Styling
ax_psd_35.set_title('PSD of Velocity Y (z = 35.0 m)')
ax_psd_35.set_ylabel('PSD [$(unit)^2$/Hz]')
ax_psd_35.grid(True, linestyle='--', linewidth=0.5)

ax_amp_35.set_title('Amplitude Spectrum of Velocity Y (z = 35.0 m)')
ax_amp_35.set_xlabel('Frequency [Hz]')
ax_amp_35.set_ylabel('Amplitude [unit]')
ax_amp_35.grid(True, linestyle='--', linewidth=0.5)

ax_psd_35.legend(fontsize='small', loc='upper right')
ax_amp_35.legend(fontsize='small', loc='upper right')
plt.tight_layout()
plt.show()


# import json
# import matplotlib.pyplot as plt
#
# json_path = r"C:\Users\ritfeldis\Documents\Python\STEM\Holten_rm_test\results\json_output_stage_2.json"
#
# with open(json_path, 'r') as f:
#     data = json.load(f)
#
# time = data['TIME']
#
# plt.figure(figsize=(10, 6))
#
# # Loop through all nodes except NODE_316
# for node_key in sorted(k for k in data.keys() if k.startswith('NODE_')):
#     coords = data[node_key]['COORDINATES']
#     velocity_z = data[node_key]['VELOCITY_Y']
#
#     label = f"{node_key} {tuple(round(c, 2) for c in coords)}"
#
#     plt.plot(time, velocity_z, label=label)
#
# plt.xlabel('Time [s]')
# plt.ylabel('Velocity Y')
# plt.title('Velocity Y over Time for Selected Nodes')
# plt.grid(True)
# plt.legend(loc='best', fontsize='small')
# plt.tight_layout()
# plt.show()

# import json
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy import signal
# from numpy.fft import rfft, rfftfreq
#
# json_path = r"C:\Users\ritfeldis\Documents\Python\STEM\Holten_rm_test\results\json_output_stage_2.json"
#
# with open(json_path, 'r') as f:
#     data = json.load(f)
#
# # Extract time and calculate sample rate
# time = np.array(data['TIME'])
# Fs = 1.0 / np.mean(np.diff(time))  # sampling frequency
# print(f"Sampling frequency: {Fs:.2f} Hz")
#
# # Create figure with two subplots
# fig, (ax_psd, ax_amp) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
#
# # Loop through all nodes except NODE_316
# for node_key in sorted(k for k in data.keys() if k.startswith('NODE_') and k != 'NODE_316'):
#     coords = data[node_key]['COORDINATES']
#     velocity_y = np.array(data[node_key]['VELOCITY_Y'])
#
#     label = f"{node_key} {tuple(round(c, 1) for c in coords)}"
#
#     # --- PSD using periodogram ---
#     freqs_psd, psd = signal.periodogram(velocity_y, fs=Fs, window='boxcar', detrend=False, scaling='spectrum')
#     ax_psd.semilogy(freqs_psd, psd, label=label)
#
#     # --- Amplitude spectrum using FFT ---
#     N = len(velocity_y)
#     freqs_amp = rfftfreq(N, d=1/Fs)
#     amp_spectrum = np.abs(rfft(velocity_y)) / N
#     ax_amp.plot(freqs_amp, amp_spectrum, label=label)
#
# # Plot styling
# ax_psd.set_title('Power Spectral Density of Velocity Y')
# ax_psd.set_ylabel('PSD [$(unit)^2$/Hz]')
# ax_psd.grid(True, which='both', linestyle='--', linewidth=0.5)
#
# ax_amp.set_title('Amplitude Spectrum of Velocity Y')
# ax_amp.set_xlabel('Frequency [Hz]')
# ax_amp.set_ylabel('Amplitude [unit]')
# ax_amp.grid(True, which='both', linestyle='--', linewidth=0.5)
#
# # Legends
# ax_psd.legend(fontsize='small', loc='upper right')
# ax_amp.legend(fontsize='small', loc='upper right')
#
# plt.tight_layout()
# plt.show()





