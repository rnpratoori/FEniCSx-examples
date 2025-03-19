# Import Libraries
from mpi4py import MPI
from dolfinx import mesh, fem, io
import numpy as np
import ufl
from dolfinx.fem.petsc import assemble_matrix, create_vector, assemble_vector, apply_lifting, set_bc
from petsc4py import PETSc
from pathlib import Path

# Define time parameters
t = 0.0  # Start time
T = 1.0  # End time
num_steps = 20  # Number of time steps
dt = T / num_steps  # Time step size

# Number of elements
N = 4000

# Create mesh
domain = mesh.create_unit_square(MPI.COMM_WORLD, N, N, cell_type=mesh.CellType.triangle)

# Create FunctionSpace
V = fem.functionspace(domain, ("Lagrange", 1))

# Print total number of DOFs
num_dofs_local = V.dofmap.index_map.size_local
num_dofs_global = domain.comm.allreduce(num_dofs_local, op=MPI.SUM)
if domain.comm.rank == 0:
    print(f"Total number of DOFs: {num_dofs_global}")

# Define initial condition
def initial_condition(x, a=5):
    return np.exp(-a * (x[0]**2 + x[1]**2))

u_n = fem.Function(V)
u_n.name = "u_n"
u_n.interpolate(initial_condition)

# Apply Dirichlet BCs
fdim = domain.topology.dim - 1
# Identify the facets on the boundaries
facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.full(x.shape[1], True, dtype=bool))
# Locate DoFs on the facets
dofs_boundary = fem.locate_dofs_topological(V, fdim, facets)
# Create Dirichlet BCs
bc = fem.dirichletbc(PETSc.ScalarType(0.0), dofs_boundary, V)

# Define solution variable
uh = fem.Function(V)
uh.name = "uh"
uh.interpolate(initial_condition)

# Define variational problem
# Define trial and test functions
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
# Define source function
x = ufl.SpatialCoordinate(domain)
f = fem.Constant(domain, PETSc.ScalarType(0))
# Define variational form
a = u * v * ufl.dx + (dt / 2) * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = ((u_n + dt * f) * v - (dt / 2) * ufl.dot(ufl.grad(u_n), ufl.grad(v))) * ufl.dx

# Assembly
bilinear_form = fem.form(a)
linear_form = fem.form(L)

# Matrix assembly with sparsity pattern
A = fem.petsc.create_matrix(bilinear_form)
A.zeroEntries()

# Assemble matrix
A = assemble_matrix(bilinear_form, bcs=[bc])
A.assemble(assembly=PETSc.Mat.AssemblyType.FINAL_ASSEMBLY)

b = create_vector(linear_form)

# Solver configuration with optimized parallel settings
solver = PETSc.KSP().create(domain.comm)
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.GMRES)  # Changed from CG to GMRES for better parallel scaling
solver.setTolerances(rtol=1e-10, max_it=1000)

# Improved AMG preconditioner settings
pc = solver.getPC()
pc.setType(PETSc.PC.Type.HYPRE)
pc.setHYPREType("boomeramg")
# Tune BoomerAMG parameters for better parallel performance
opts = PETSc.Options()
opts.setValue("pc_hypre_boomeramg_strong_threshold", 0.5)
opts.setValue("pc_hypre_boomeramg_agg_nl", 2)
opts.setValue("pc_hypre_boomeramg_coarsen_type", "HMIS")
opts.setValue("pc_hypre_boomeramg_interp_type", "ext+i")
pc.setFromOptions()

# Time-stepping
for i in range(num_steps):
    t += dt
    # Update the right-hand side reusing the initial vector
    with b.localForm() as loc_b:
        loc_b.set(0)
    assemble_vector(b, linear_form)
    # Apply Dirichlet boundary condition to the vector
    apply_lifting(b, [bilinear_form], [[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b, [bc])
    # Solve linear problem
    solver.solve(b, uh.x.petsc_vec)
    uh.x.scatter_forward()
    # Update solution at previous time step (u_n)
    u_n.x.array[:] = uh.x.array