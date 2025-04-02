import numpy as np
import matplotlib.pyplot as plt

# Plate and Soil Parameters
a, b = 10, 10            # Plate dimensions (m)
h = 0.1                  # Plate thickness (m)  100 mm
E = 200e9                # Young's Modulus of Plate (Pa)
nu = 0.3                 # Poisson's ratio of Plate

# Soil Properties
Es = 50e1                # Young's Modulus of Soil (Pa)
nu_s = 0.3               # Poisson's ratio of Soil
hs = 2                   # Depth of soil domain (m)

# Calculating Equivalent Subgrade Reaction Modulus
K = (Es / (1 - nu_s**2)) * (a * b) / hs

# Flexural rigidity
D = (E * h**3) / (12 * (1 - nu**2))

# Deflection function
def plate_deflection(x, y, a, b, C):
    R = x / a
    Q = y / b
    return C * ((1 - R)**2 * R**2) * ((1 - Q)**2 * Q**2)

# Calculate deflection coefficient
def calculate_deflection_coefficient(D, q, a, b, K):
    return (q * a**4) / (64 * D + K * a**4)

# Parameters for Load
q = 1000  # Uniform Load in N/m^2 (Pa)

# Deflection calculation
C = calculate_deflection_coefficient(D, q, a, b, K)

# Creating Grid
x = np.linspace(0, a, 100)
y = np.linspace(0, b, 100)
X, Y = np.meshgrid(x, y)
W = plate_deflection(X, Y, a, b, C)

# get the maximum deflection
max_deflection = np.max(W)
print(f"Maximum deflection: {max_deflection} m")

# Plotting the deflection
plt.figure(figsize=(8, 6))
plt.contourf(X, Y, W, levels=30, cmap='viridis')
plt.colorbar(label='Deflection (m)')
plt.title('Deflection of Clamped Plate on 3D Soil Domain (Equivalent Winkler Model)')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.grid(True)
plt.show()
