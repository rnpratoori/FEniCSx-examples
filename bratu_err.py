# Import Libraries
from mpi4py import MPI
from dolfinx import mesh, fem, default_scalar_type
import numpy as np
import ufl
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from petsc4py import PETSc
from pathlib import Path
import matplotlib.pyplot as plt

# Define function to be used for the source term and analytical solution
def u_ex(mod):
    return lambda x: mod.sin(mod.pi * x[0]) * mod.sin(mod.pi * x[1])

u_numpy = u_ex(np)
u_ufl = u_ex(ufl)

# Define function to create different problem instances for different meshes
def solve_bratu(N):
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
    u = fem.Function(V)
    v = ufl.TestFunction(V)
    w = ufl.TrialFunction(V)
    x = ufl.SpatialCoordinate(domain)
    lambda_ = fem.Constant(domain, PETSc.ScalarType(1.0))
    # Compute the manufactured source term
    f = -ufl.div(ufl.grad(u_ufl(x))) - lambda_ * ufl.exp(u_ufl(x))
    # Define residual and Jacobian
    F = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - lambda_ * ufl.exp(u) * v * ufl.dx - ufl.inner(f, v) * ufl.dx
    J = ufl.inner(ufl.grad(w), ufl.grad(v)) * ufl.dx - lambda_ * ufl.exp(u) * w * v * ufl.dx

    # Create Nonlinear Problem
    problem = NonlinearProblem(F, u, [bc], J)
    solver = NewtonSolver(domain.comm, problem)
    solver.atol = 1e-10
    solver.rtol = 1e-10
    solver.max_it = 50
    solver.solve(u)
    
    # Interpolate exact solution
    u_exact = fem.Function(V)
    u_exact.interpolate(u_numpy)

    return u, u_exact

# Define function to compute error at a higher degree
def error_L2(uh, u_ex, degree_raise=3):
    # Create higher order function space
    degree = uh.function_space.ufl_element().degree
    family = uh.function_space.ufl_element().family_name
    mesh = uh.function_space.mesh
    W = fem.functionspace(mesh, (family, degree + degree_raise))
    # Interpolate approximate solution
    u_W = fem.Function(W)
    u_W.interpolate(uh)

    # Interpolate exact solution, special handling if exact solution
    # is a ufl expression or a python lambda function
    u_ex_W = fem.Function(W)
    if isinstance(u_ex, ufl.core.expr.Expr):
        u_expr = fem.Expression(u_ex, W.element.interpolation_points())
        u_ex_W.interpolate(u_expr)
    else:
        u_ex_W.interpolate(u_ex)

    # Compute the error in the higher order function space
    e_W = fem.Function(W)
    e_W.x.array[:] = u_W.x.array - u_ex_W.x.array

    # Integrate the error
    error = fem.form(ufl.inner(e_W, e_W) * ufl.dx)
    error_local = fem.assemble_scalar(error)
    error_global = mesh.comm.allreduce(error_local, op=MPI.SUM)
    return np.sqrt(error_global)

# Solve problem for different mesh resolutions and compute errors
N_values = [2**i for i in range(2, 10)]
Errors = np.zeros(len(N_values), dtype=default_scalar_type)
h_values = np.zeros(len(N_values), dtype=np.float64)

for i, N in enumerate(N_values):
    u_h, u_ex = solve_bratu(N)
    comm = u_h.function_space.mesh.comm
    # error = fem.form((u_h - u_ex)**2 * ufl.dx)
    Errors[i] = error_L2(u_h, u_numpy)
    h_values[i] = 1 / N_values[i]
    if comm.rank == 0:
        print(f"h: {h_values[i]:.2e} Error: {Errors[i]:.2e}")
rates = np.log(Errors[1:] / Errors[:-1]) / np.log(h_values[1:] / h_values[:-1])
if comm.rank == 0:
    print(f"Polynomial degree 1, Rates {rates}")

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
    plt.title('Error Convergence - Bratu')
    plt.savefig('bratu_err.png')