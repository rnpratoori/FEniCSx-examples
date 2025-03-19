# Import Libraries
from mpi4py import MPI
from dolfinx import mesh, fem, default_scalar_type, log
import numpy as np
import ufl
from dolfinx.fem.petsc import assemble_matrix, create_vector, assemble_vector, apply_lifting, set_bc
from petsc4py import PETSc
from pathlib import Path
import matplotlib.pyplot as plt
import time
import tracemalloc
from memory_profiler import profile
import cProfile, pstats

# Define function to be used for the source term and analytical solution
def u_ex(mod):
    return lambda x, t: mod.exp(-t) * mod.sin(mod.pi * x[0]) * mod.sin(mod.pi * x[1])

# Define function for source term
def f_numpy(x, t):
    return (2 * np.pi**2 - 1) * np.exp(-t) * np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])

u_numpy = u_ex(np)
u_ufl = u_ex(ufl)

# Define function to create different problem instances for different meshes
# @profile
def solve_tdht(N=500, num_steps=100):
    if MPI.COMM_WORLD.rank==0:
        print(f"*****Running simulation for N={N}, num_steps={num_steps}******")
    # Define time parameters
    t = 0.0  # Start time
    t_n = 0.0 # Previous time
    T = 0.125  # End time
    dt = T / num_steps  # Time step size

    # Create mesh
    domain = mesh.create_unit_square(MPI.COMM_WORLD, N, N)

    # Create FunctionSpace
    V = fem.functionspace(domain, ("Lagrange", 1))

    # Define initial condition
    u_n = fem.Function(V)
    u_n.name = "u_n"
    u_n.interpolate(lambda x: u_numpy(x,0.0))

    # Apply Dirichlet BCs
    # Identify the facets on the left and right boundaries
    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.full(x.shape[1], True))
    # Locate DoFs on the facets
    dofs_boundary = fem.locate_dofs_topological(V, fdim, facets)
    # Create Dirichlet BCs

    # Define solution variable
    uh = fem.Function(V)
    uh.name = "uh"
    uh.x.array[:] = u_n.x.array[:]

    # Define variational problem
    # Define trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    # Define source function
    x = ufl.SpatialCoordinate(domain)
    # Define source function
    f = fem.Function(V)
    # Define variational form
    a = u * v * ufl.dx + (dt / 2) * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
    bilinear_form = fem.form(a)
    
    u_bc = fem.Function(V)

    # Create solver
    solver = PETSc.KSP().create(domain.comm)
    solver.setType(PETSc.KSP.Type.PREONLY)
    solver.getPC().setType(PETSc.PC.Type.LU)
    # option_prefix = solver.getOptionsPrefix()

    # tracemalloc.start()
    
    # Time-stepping
    for i in range(num_steps):
        # step_profiler = cProfile.Profile()
        # step_profiler.enable()

        t += dt

        # Update BC
        u_bc.interpolate(lambda x: u_numpy(x,t=t))
        bc = fem.dirichletbc(u_bc, dofs_boundary)

        # Update source term
        f.interpolate(lambda x: (f_numpy(x, t) + f_numpy(x, t_n)) / 2)
        L = ((u_n + dt * f) * v - (dt / 2) * ufl.dot(ufl.grad(u_n), ufl.grad(v))) * ufl.dx

        # t0 = time.time()
        # Assembly
        linear_form = fem.form(L)
        A = assemble_matrix(bilinear_form, bcs=[bc])
        A.assemble()
        b = create_vector(linear_form)
        with b.localForm() as loc_b:
            loc_b.set(0)
        assemble_vector(b, linear_form)
        # print(f"Assembly time: {time.time() - t0:.4f} seconds")
        
        # t0 = time.time()
        # Apply Dirichlet boundary condition to the vector
        apply_lifting(b, [bilinear_form], [[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b, [bc])
        # Solve linear problem
        solver.setOperators(A)
        solver.solve(b, uh.x.petsc_vec)
        # print(f"Solve time: {time.time() - t0:.4f} seconds")
        
        uh.x.scatter_forward()
        # Update solution at previous time step (u_n)
        u_n.x.array[:] = uh.x.array
        t_n = t

        # step_profiler.disable()
        # step_profiler.print_stats(sort='cumtime')
    # snapshot = tracemalloc.take_snapshot()
    # top_stats = snapshot.statistics("lineno")
    # if MPI.COMM_WORLD.rank == 0:
    #     print(f"[Step {i}] Top Memory Allocations")
    #     for stat in top_stats[:5]:  # Print top 5 allocations
    #         print(stat)
    # tracemalloc.stop()
    
    # Interpolate exact solution
    u_exact = fem.Function(V)
    u_exact.interpolate(lambda x: u_numpy(x, T))
    
    return uh, u_exact

if __name__ == '__main__':

    # Solve problem for different number of timesteps and compute errors
    num_steps_values = [2**i for i in range(0, 7)]
    # num_steps_values = [100*i for i in range(1, 11)]
    t_Errors = np.zeros(len(num_steps_values), dtype=default_scalar_type)
    dt_values = np.zeros(len(num_steps_values), dtype=np.float64)

    for i, num_steps in enumerate(num_steps_values):
        if MPI.COMM_WORLD.rank == 0:
            profiler = cProfile.Profile()
            profiler.enable()
        u_h, u_ex = solve_tdht(num_steps=num_steps)
        comm = u_h.function_space.mesh.comm
        # t_Errors[i] = error_L2(u_h, u_numpy)
        error = fem.form((u_h - u_ex)**2 * ufl.dx)
        t_Errors[i] = np.sqrt(comm.allreduce(fem.assemble_scalar(error), MPI.SUM))
        dt_values[i] = 0.125 / num_steps_values[i]
        if comm.rank == 0:
            print(f"dt: {dt_values[i]:.2e} Error: {t_Errors[i]:.2e}")
        if MPI.COMM_WORLD.rank == 0:
            profiler.disable()
            pstats.Stats(profiler).sort_stats("cumtime").print_stats(20)
    rates = np.log(t_Errors[1:] / t_Errors[:-1]) / np.log(dt_values[1:] / dt_values[:-1])
    if comm.rank == 0:
        print(f"Polynomial degree 1, Rates {rates}")

# Generate convergence plot for timestep size
if MPI.COMM_WORLD.rank == 0:
    plt.figure()
    plt.loglog(dt_values, t_Errors, marker='o', linestyle='-', label='L2 Error')
    # Add reference line with slope 1
    slope = 2 
    ref_line = t_Errors[0] * (dt_values / dt_values[0])**slope
    plt.loglog(dt_values, ref_line, linestyle='--', label=f'Ref: slope {slope}')
    plt.xlabel('Time step size (dt)')
    plt.ylabel('L2 Error')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.title('Error Convergence - Time dependent heat')
    plt.savefig('tdht_err_t.png')
