# Import Libraries
from mpi4py import MPI
from dolfinx import mesh, fem, plot, io, log
import numpy as np
import ufl
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from petsc4py import PETSc
import pyvista as pv
from pathlib import Path

# Create mesh
domain = mesh.create_unit_square(MPI.COMM_WORLD, 50, 50, cell_type=mesh.CellType.triangle)

# Create FunctionSpace
V = fem.functionspace(domain, ("Lagrange", 1))

# Apply Dirichlet BCs
# Identify the facets on the left and right boundaries
facets = mesh.locate_entities_boundary(domain, domain.topology.dim - 1, lambda x: np.full(x.shape[1], True, dtype=bool))
# Locate DoFs on the facets
dofs_boundary = fem.locate_dofs_topological(V, domain.topology.dim - 1, facets)
# Create Dirichlet BCs
bc = fem.dirichletbc(PETSc.ScalarType(0.0), dofs_boundary, V)

# Define variational problem
# Define trial and test functions
u = fem.Function(V)
v = ufl.TestFunction(V)
w = ufl.TrialFunction(V)
x = ufl.SpatialCoordinate(domain)
lambda_ = fem.Constant(domain, PETSc.ScalarType(1.0))
# Define residual
F = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - lambda_ * ufl.exp(u) * v * ufl.dx
# Define Jacobian
J = ufl.inner(ufl.grad(w), ufl.grad(v)) * ufl.dx - lambda_ * ufl.exp(u) * w * v * ufl.dx

# Create NonlinearProblem
problem = NonlinearProblem(F, u, [bc], J)

# Create Newton Solver
solver = NewtonSolver(domain.comm, problem)
# Set solver options
solver.atol = 1e-8
solver.rtol = 1e-7
solver.max_it = 50

# Solve nonlinear problem
log.set_log_level(log.LogLevel.INFO)
n, converged = solver.solve(u)

# Define residual
residual = fem.Form(F)
jacobian = fem.Form(J)
if converged:
    print(f"Newton solver converged in {n} iterations.")
else:
    print("Newton solver did NOT converge!")

# Post-process
# Save solution to file
results_folder = Path("results")
results_folder.mkdir(exist_ok=True, parents=True)
filename = results_folder / "out_bratu"
with io.XDMFFile(domain.comm, filename.with_suffix(".xdmf"), "w", io.XDMFFile.Encoding.ASCII) as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(u)

