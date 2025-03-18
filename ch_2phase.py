# Import Libraries
from mpi4py import MPI
from dolfinx import mesh, fem, io, log, default_real_type
import numpy as np
from script import create_periodic_mesh
import ufl
from basix.ufl import element, mixed_element
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from petsc4py import PETSc
from pathlib import Path

# Define simulation parameters
lambda_ = 1.0e-2
dt = 5.0e-06
T = 1.0e-03 # End time
num_steps = T / dt # Number of time steps

# Create mesh
domain = mesh.create_unit_square(MPI.COMM_WORLD, 100, 100, cell_type=mesh.CellType.triangle)

# Apply Periodic BCs
# Extract the dimension of the mesh
L_min = [domain.comm.allreduce(np.min(domain.geometry.x[:, i]), op=MPI.MIN) for i in range(3)]
L_max = [domain.comm.allreduce(np.max(domain.geometry.x[:, i]), op=MPI.MAX) for i in range(3)]
# Define the periodic boundary condition
def i_x(x):
    return np.isclose(x[0], L_min[0])

def i_y(x):
    return np.isclose(x[1], L_min[1])

def indicator(x):
    return i_x(x) | i_y(x)

def mapping(x):
    values = x.copy()
    values[0] += i_x(x) * (L_max[0] - L_min[0])
    values[1] += i_y(x) * (L_max[1] - L_min[1])
    return values

# domain, replaced_vertices, replacement_map = create_periodic_mesh(domain, indicator, mapping)
fdim = domain.topology.dim - 1
domain.topology.create_entities(fdim)
domain, replaced_vertices, replacement_map = create_periodic_mesh(domain, indicator, mapping)
domain.topology.create_entities(fdim)
# Locate facets for boundary conditions and create meshtags
domain.topology.create_connectivity(fdim, fdim + 1)

# Create FunctionSpace
P1 = element("Lagrange", domain.basix_cell(), 1, dtype=default_real_type)
ME = fem.functionspace(domain, mixed_element([P1, P1]))

# Define variational problem
# Define trial and test functions
u = fem.Function(ME)
# Previous solution
u0 = fem.Function(ME)
q, v = ufl.TestFunctions(ME)
# Split mixed functions
c, mu = ufl.split(u)
c0, mu0 = ufl.split(u0)

# Initial condition
u.x.array[:] = 0.0
rng = np.random.default_rng(42)
u.sub(0).interpolate(lambda x: 0.63 + 0.02 * (0.5 - rng.random(x.shape[1])))
u.x.scatter_forward()

# Compute df/dc
c = ufl.variable(c)
f = 100 * c**2 * (1 - c) ** 2
dfdc = ufl.diff(f, c)

# Define residuals
F0 = ufl.inner(c, q) * ufl.dx - ufl.inner(c0, q) * ufl.dx + (dt/2) * ufl.inner(ufl.grad(mu + mu0), ufl.grad(q)) * ufl.dx
F1 = ufl.inner(mu, v) * ufl.dx - ufl.inner(dfdc, v) * ufl.dx - lambda_ * ufl.inner(ufl.grad(c), ufl.grad(v)) * ufl.dx
F = F0 + F1

# Create NonlinearProblem
problem = NonlinearProblem(F, u)

# Create Newton Solver
# log.set_log_level(log.LogLevel.INFO)
solver = NewtonSolver(domain.comm, problem)
solver.convergence_criterion = "incremental"
solver.rtol = 1.0e-6
ksp = solver.krylov_solver
opts = PETSc.Options()  # type: ignore
option_prefix = ksp.getOptionsPrefix()
opts[f"{option_prefix}ksp_type"] = "preonly"
opts[f"{option_prefix}pc_type"] = "lu"
sys = PETSc.Sys()
opts[f"{option_prefix}pc_factor_mat_solver_type"] = "superlu_dist"
ksp.setFromOptions()

# Post-process
# Save solution to file
t = 0.0
results_folder = Path("results")
results_folder.mkdir(exist_ok=True, parents=True)
filename = results_folder / "out_ch2p_pbc.bp"
with io.VTXWriter(domain.comm, filename, [u.sub(0)], engine="BP4") as vtx:
    vtx.write(t)
    # Time-stepping
    c = u.sub(0)
    u0.x.array[:] = u.x.array
    while t < T:
        t += dt
        r = solver.solve(u)
        print(f"Step {int(t / dt)}: num iterations: {r[0]}")
        u0.x.array[:] = u.x.array
        vtx.write(t)