# Import Libraries
from mpi4py import MPI
from dolfinx import mesh, fem, default_scalar_type, io
import numpy as np
import ufl
from dolfinx.fem.petsc import LinearProblem
from pathlib import Path

# Create mesh
domain = mesh.create_rectangle(MPI.COMM_WORLD, ((0.0, 0.0), (2.0, 1.0)), (20, 10), cell_type=mesh.CellType.quadrilateral)

# Create FunctionSpace
V = fem.functionspace(domain, ("Lagrange", 1))

# Apply Dirichlet BCs
# Identify the facets on the left and right boundaries
facets = mesh.locate_entities_boundary(domain, domain.topology.dim - 1, lambda x: np.isclose(x[0], 0.0) | np.isclose(x[0], 2.0))
# Locate DoFs on the facets
dofs_boundary = fem.locate_dofs_topological(V, domain.topology.dim - 1, facets)
# Create Dirichlet BCs
bc = fem.dirichletbc(default_scalar_type(0.0), dofs_boundary, V)

# Define variational problem
# Define trial and test functions
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
# Define source function
x = ufl.SpatialCoordinate(domain)
f = 10.0 * ufl.exp(- ((x[0] - 0.5)**2 + (x[1] - 0.5)**2) / 0.02)
# Define variational form
a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = ufl.inner(f, v) * ufl.dx

# Solve variational problem
problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()

# Post-process
# Save solution to file
results_folder = Path("results")
results_folder.mkdir(exist_ok=True, parents=True)
filename = results_folder / "out_ssht.bp"
with io.VTXWriter(domain.comm, filename, [uh], engine="BP4") as vtx:
    vtx.write(0.0)
