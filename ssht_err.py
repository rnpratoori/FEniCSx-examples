# Import Libraries
from mpi4py import MPI
from dolfinx import mesh, fem, default_scalar_type
import numpy as np
import ufl
from dolfinx.fem.petsc import LinearProblem
from pathlib import Path
import matplotlib.pyplot as plt

# Define function to be used for the source term and analytical solution
def u_ex(mod):
    return lambda x: mod.sin(mod.pi * x[0]) * mod.sin(mod.pi * x[1])

u_numpy = u_ex(np)
u_ufl = u_ex(ufl)

# Define function to create different problem instances for different meshes
def solve_ssht(N):
    # Create mesh
    domain = mesh.create_unit_square(MPI.COMM_WORLD, N, N)

    # Create FunctionSpace
    V = fem.functionspace(domain, ("Lagrange", 1))

    # Apply Dirichlet BCs
    # Identify the facets on the left and right boundaries
    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.full(x.shape[1], True))
    # Locate DoFs on the facets
    dofs_boundary = fem.locate_dofs_topological(V, fdim, facets)
    # Create Dirichlet BCs
    u_bc = fem.Function(V)
    u_bc.interpolate(u_numpy)
    bc = fem.dirichletbc(u_bc, dofs_boundary)

    # Define variational problem
    # Define trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    # Define source function
    x = ufl.SpatialCoordinate(domain)
    f = -ufl.div(ufl.grad(u_ufl(x)))
    # Define variational form
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx

    # Solve variational problem
    problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    
    return problem.solve(), u_ufl(x)

# Solve problem for different mesh resolutions and compute errors
N_values = [2**i for i in range(3, 10)]
Errors = np.zeros(len(N_values), dtype=default_scalar_type)
h_values = np.zeros(len(N_values), dtype=np.float64)

for i, N in enumerate(N_values):
    u_h, u_ex = solve_ssht(N)
    comm = u_h.function_space.mesh.comm
    error = fem.form((u_h - u_ex)**2 * ufl.dx)
    Errors[i] = np.sqrt(comm.allreduce(fem.assemble_scalar(error), MPI.SUM))
    h_values[i] = 1 / N_values[i]

# Generate convergence plot
if MPI.COMM_WORLD.rank == 0:
    plt.figure()
    plt.loglog(h_values, Errors, marker='o', linestyle='-', label='L2 Error')
    # Add reference line with slope 2
    slope = 2
    ref_line = Errors[0] * (h_values / h_values[0])**slope
    plt.loglog(h_values, ref_line, linestyle='--', label=f'Ref: slope {slope}')
    plt.xlabel('Mesh size (h)')
    plt.ylabel('L2 Error')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.title('Error Convergence')
    plt.savefig('ssht_err.png')