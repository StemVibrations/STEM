import numpy as np
import matplotlib.pyplot as plt
from UVEC.uvec_ten_dof_vehicle_2D.irregularities import calculate_rail_irregularity

# Define range of x values over 10 meters
x_values = np.linspace(0, 10, 1000)
y_irregularities = [calculate_rail_irregularity(x) for x in x_values]

# Plot
plt.figure(figsize=(12, 4))
plt.plot(x_values, y_irregularities, label='Rail Unevenness')
plt.xlabel('Position along rail [m]')
plt.ylabel('Unevenness [m]')
#plt.title('Rail Unevenness over 10 Meter Track')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(r"C:\Users\ritfeldis\Documents\Python\STEM\railuneveness_example.png")
plt.show()