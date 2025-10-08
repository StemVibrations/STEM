# This are the measurements of Holten. The Z-direction is the height direction, this is for us interesting.
# There are measurements of VIRM and SPR(A) (intercity and sprinter)
# This are the results of track 1 (naar Rijssen)
# Speeds are measured (mm/s)

# import pickle
#
# # Data laden
# with open("D:/VIRM.pkl", "rb") as f:
#     data = pickle.load(f)
#
# def print_keys(d, indent=0):
#     """Recursief alle keys in een dict printen met inspringing."""
#     if isinstance(d, dict):
#         for k, v in d.items():
#             print("  " * indent + str(k))
#             print_keys(v, indent + 1)
#
# print_keys(data)

import pickle

# Data laden
with open("D:/VIRM.pkl", "rb") as f:
    data = pickle.load(f)

# Kies een meetpunt en tijd
meetpunt = 'Meetjournal_MP8_Holten_zuid_4m_C'
tijdstip = '2024-09-10 14:16:06'

info = data[meetpunt][tijdstip]

print("Datum:", info["date"])
print("Treintype:", info["train_type"])
print("Spoor (track):", info["track"])
print("Snelheid [km/h]:", info["speed"])   # meestal in km/h opgeslagen


# import pickle
# import numpy as np
# import matplotlib.pyplot as plt
#
# # Data laden
# with open("D:/VIRM.pkl", "rb") as f:   # kleine fix: gebruik forward slash of dubbele backslash
#     data = pickle.load(f)
#
# meetpunten = list(data.keys())
# print(meetpunten)
#
# # Selecteer meetpunt en tijd
# # meetpunt = 'Meetjournal_MP4_Holten_zuid_2m_C'
# # meetpunt = 'Meetjournal_MP8_Holten_zuid_4m_C'
# # meetpunt = 'Meetjournal_MP1_Holten_zuid_16m_C'
# meetpunt = 'Meetjournal_MP2_Holten_zuid_25m_C'
#
# # This looks like other locations then given in the rapport? Is 23m, 25m?
# meetpunt = 'Meetjournal_MP2_Holten_zuid_25m_C'
# tijdstippen = list(data[meetpunt].keys())
#
# print(f"Beschikbare tijdstippen/traces voor {meetpunt}:")
# for t in tijdstippen:
#     print(t)
#
# tijdstip = '2024-09-10 14:16:06'
#
# time = data[meetpunt][tijdstip]['data']['TIME']
# trace_z = np.array(data[meetpunt][tijdstip]['data']['TRACE_Z']) / 1000.0  # naar m/s
#
# # Plot
# plt.figure(figsize=(10, 5))
# plt.plot(time, trace_z, color="steelblue", linewidth=1)
#
# # Labels & titel
# plt.xlabel("Tijd [s]")
# plt.ylabel("Trillingssnelheid Z [m/s]")   # of mm/s als de data zo genormaliseerd is
# plt.ylim([-0.006, 0.006])
# plt.title(f"Tijdsignaal {meetpunt} ({tijdstip})")
# plt.grid(True, linestyle="--", alpha=0.6)
#
# plt.tight_layout()
# plt.show()
