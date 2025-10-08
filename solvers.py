import inspect
from dataclasses import is_dataclass
import stem.solver

solver_names = []

for name, obj in inspect.getmembers(stem.solver):
    # Check of het een dataclass en subclass is van LinearSolverSettingsABC
    if inspect.isclass(obj) and is_dataclass(obj):
        bases = [base.__name__ for base in obj.__bases__]
        if "LinearSolverSettingsABC" in bases:
            instance = obj()
            solver_names.append(instance.solver_type)  # alleen de solver_type

# Print alle gevonden solver_type namen
print("Beschikbare lineaire solvers:")
for solver_type in solver_names:
    print(solver_type)



