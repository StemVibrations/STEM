import numpy as np

E = 200e6
nu = 0.495
rho = 1900
f = 100

M = E * (1 - nu) / ((1 + nu) * (1 - 2 * nu))
vp = np.sqrt(M / rho)
lambda_ = vp / f
required_el_size = lambda_ / 10
required_dt = required_el_size / vp

print(f"{vp:.2f}")
print(f"{lambda_:.2f}")
print(f"{required_el_size:.3f}")
print(f"{required_dt:.6f}")
