"""
Spinodal decomposition model by Souzan Hammadi & Nana Ofori-Opoku

Fields in F:

0. 𝔽: function space
1. 𝐹: Helmholtz free energy
2. x₁: composition of c1
3. |∇c|²: squared gradient of concentration
4. f(c): thermodynamic energy density

Read input parameters & settings from `config.py`
Output runtime values to `record.yml`
"""
from __future__ import print_function

# === Basic performance monitoring ===

import time

startTime = time.time()

# === System libraries ===
import gc
import numpy as np
import os
import pandas as pd
import ruamel.yaml

# import steppyngstounes as stp
import sys

# === Dolfin/FEniCS ===

import ufl as ufl
import dolfin as df

# === AMR and other secret sauce ===

import amr_fenics.input_output as io
import amr_fenics.mesh_domain_funcs as mdf
import amr_fenics.solver_funcs as slf
import amr_fenics.helpers as hlp
import amr_fenics.mesh_refinement as amr
import amr_fenics.utility as ut

# === Job Parameters ==
from config import Adapt, Domain, Interval, Model, Solver, PetscOpt

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OMPI_ALLOW_RUN_AS_ROOT"] = "1"
pd.set_option("display.width", 1000)
pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", "{:.4f}".format)

run = "." if len(sys.argv) < 2 else sys.argv[1]
restartFlag = os.path.isfile("{}/restart.h5".format(run)) #==True

if restartFlag:
    restart_run = sys.argv[2]
    fs = io.output_directory(restart_run, "solution")
    fs_fine = io.output_directory(restart_run, "fine_solution")
    nrg_file = restart_run + "/" + "ene.csv"
else:
    fs =  io.output_directory(run, "solution")
    fs_fine = io.output_directory(run, "fine_solution")
    nrg_file = run + "/" + "ene.csv"

names = [r"$c$", r"$\mu$"]
auxnames = [r"$\mathcal{F}$", r"$\Delta c$"]

record = {"data": {}, "domain": {}, "model": {}, "runtime": {}, "adapt": {}}

stats_head = [
    "time",  # "simulation time, dimensionless"
    "dt", #timestep, dimensionless
    "c",  # "Li composition, whole domain"
    "μ",  # average chemical potential
    "Helmholtz",  # "Helmholtz free energy, ∫F·dx"
    "wall_time",  # "runtime (sec)"
    #"mem_GB",  # "current memory usage (GB)"
    #"peak_GB",  # "maximum memory usage (GB)"
]

record["data"]["stats"] = {}
record["data"]["stats"]["file"] = nrg_file
record["data"]["stats"]["cols"] = stats_head

# === Runtime configuration ===

# Solver options
BDF2 = Solver["BDF2"]
dt = df.Constant(Interval["timestep"])  # time step
dt_max = Interval["maxtimestep"]  # maximum time step
t_interval_output = Interval["time_interval_output"]  # time interval output
cback = Interval["cutback"]  # reduction of time step
growth = Interval["growth"]  # growth of time step
min_its = Interval["min_iter"]  # minimum nonlinear iterations
max_its = Interval["max_iter"]  # maximum nonlinear iterations
checkpoint_frequency = Interval["f_checkpoint"]  # checkpoint frequency

D11 = record["model"]["D11"] = float(Model["D11"])  # diff coeff [m²/s]

σ = record["model"]["σ"] = Model["sigma"]  # surface energy
Wc = record["model"]["Wc"] = Model["Wc"]  # length scale

tc = record["model"]["tc"] = Wc**2 / D11  # time scale
To = record["model"]["To"] = Model["To"]  # temperature
vm = record["model"]["vm"] = Model["vm"]  # molar volume
R = record["model"]["R"] = Model["R"]  # gas constant
RTv = R * To / vm
Om = record["model"]["Ω"] = Model["Omega"]  # enthalpy of mixing coeff

Δf = -RTv * df.ln(2) + Om / vm / 4  # energy barrier scale [J/m³]

H = record["model"]["H"] = σ / Wc  # energy scale

E = record["model"]["E"] = Model["E"]  # Young's modulus
ν = record["model"]["ν"] = Model["ν"]  # Poisson's ratio
n = record["model"]["n"] = Model["n"]  # linear expansion coefficient
co = record["model"]["co"] = Model["co"]  # average concentration

Em = E / H

# === Non-dimensionalize model parameters ===
T = record["runtime"]["tfinal"] = Interval["runtime"] / tc  # total time (max)

io.disp("Critical & controlling parameters:")
io.disp(f"     Δf={Δf:<12g}    H={H:<12g}")
io.disp(f"     D11={D11:<12g}")
io.disp(f"     t={tc:<12g}     T={T:<12g}")

# === Define the mesh, size & resolution ===

"""
The program starts with a coarse mesh, and refines it — dividing the
element size by two — until the target element size is reached.
A heuristic sense is that 5 to 8 levels of refinement delivers a
good mesh, meaning the starting point (emax) depends on the goal (nde):

    emax = nde * 2ʳ⁻¹ = 64 * nde where r is the resolution in this case 7
"""
emax = record["domain"]["emax"] = Domain["emax"]
Lx = record["domain"]["Lx"] = Domain["Lx"]  # domain length
Ly = record["domain"]["Ly"] = Domain["Ly"]  # domain width
Lz = record["domain"]["Lz"] = Domain["Lz"]  # domain width

Nx = record["domain"]["Nx"] = int(Lx / emax)  # number of initial elements x-dir
Ny = record["domain"]["Ny"] = int(Ly / emax)  # number of initial elements y-dir

if Lz is not None:
    Nz = record["domain"]["Ny"] = int(Lz / emax)  # number of initial elements y-dir

emin = record["domain"]["emin"] = Domain["nde"]

io.disp(f"     smallest element, cartesian distance(i.e. Δx)={emin}")

# dimensionless entropy of mixing coefficient
RTv = record["model"]["RTv"] = RTv / H

# dimensionless enthalphy of mixing
Om = Om / vm / H

# Diffusion coefficient
D11m = record["model"]["D11m"] = D11 * (tc / Wc**2)  # dimensionless

io.disp(f"     RTv={RTv:<12g}")
io.disp(f"     D11m={D11m:<11g}")

# setting some backend parameters and PETSc solver options directions
slf.set_backend(Domain, PetscOpt)

# Class representing the intial conditions
class InitialConditions(df.UserExpression):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        np.random.seed(19960517)

    def eval(self, values, x):
        c = 0.5 + 0.1 * (0.5 - np.random.random())
        values[0] = c    # concentration
        values[1] = 0.0  # chemical potential

    def value_shape(self):
        return(2,)

def plog(c, tol):
    Δc = c - tol

    t0 = df.ln(tol)
    t1 = Δc / tol
    t2 = -(Δc**2) / (2 * tol**2)
    t3 = Δc**3 / (3 * tol**3)

    return ufl.conditional(ufl.lt(c, tol), t0 + t1 + t2 + t3, df.ln(c))

def fchem(rtv, om, c):
    x1 = c
    x2 = 1 - c
    entropy = rtv * (x1 * plog(x1, 1e-3) + x2 * plog(x2, 1e-3))
    enthalpy = om * x1 * x2

    return entropy + enthalpy

def fcahn(Em, n, ν, co, c):
    fcahn = ((Em * n**2) / (1 - ν)) * (c - co)**2
    return fcahn

def equation_solve(U, U0, ME, mesh, element, dt, BDF2, time_bdf2):
    # Split mixed functions
    c, mu = df.split(U)
    c0, mu0 = df.split(U0[0])

    if BDF2:
        c00, mu00 = df.split(U0[1])

    q, v = df.TestFunctions(ME)

    nME = df.FunctionSpace(mesh, element)
    cv = df.Function(nME)
    c_v = df.variable(cv)

    replacement = {cv: c}

    fdens = fchem(RTv, Om, c_v) + fcahn(Em, n, ν, co, c_v)

    #dfdc = df.diff(fdens, c_v)
    #dμdc = ufl.replace(df.diff(dfdc, c_v), replacement)

    dμdc = (RTv / (c * (1 - c))) - (2 * Om) + ((2 * Em * n**2) / (1 - ν))

    M = D11m / df.sqrt(dμdc**2 + 1e-6)

    fgrad = (1 / 2) * df.inner(df.grad(c_v), df.grad(c_v))
    
    Func = fgrad + fdens

    μₘ = (1.0 - Solver["theta"]) * mu0 + Solver["theta"] * mu  # mu_(n+theta)
    # Weak statement of the equations

    # Linear equation for c -> ∫ ∂c/∂t vₜₑₛₜ  = D(c)∇μ⋅∇vₜₑₛₜ + JBV⋅n̂ vₜₑₛₜ
    rhs_c = -M * df.inner(df.grad(μₘ), df.grad(q)) * df.dx

    if BDF2:
        L_c = slf.set_bdf2(c, c0, c00, q, time_bdf2, rhs_c)
    else:
        L_c = slf.set_bdf1(c, c0, q, dt, rhs_c)

    # Linear equation for mu -> ∫ μ*vₜₑₛₜ = δF/δc⋅vₜₑₛₜ
    dF = df.derivative(Func, cv, v)
    rhs_mu = ufl.replace(dF, replacement) * df.dx

    L_mu = mu * v * df.dx - rhs_mu

    L = L_c + L_mu

    fgradc = df.sqrt(df.inner(df.grad(c), df.grad(c)))

    Func = ufl.replace(Func, replacement)

    return L, Func, fgradc

def boundaries(mesh, V, bcs):
    # BOUNDARY CONDITIONS
    boundaries = df.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundaries.set_all(0)

    # 100 -  surfaces
    left = mdf.xboundary(0)
    right = mdf.xboundary(Lx)

    left.mark(boundaries, 1)
    right.mark(boundaries, 2)

    # 010 - surfaces
    bottom = mdf.yboundary(0)
    top = mdf.yboundary(Ly)

    bottom.mark(boundaries, 3)
    top.mark(boundaries, 4)

    # 001 - surfaces
    if Lz is not None:
        back = mdf.zboundary(0)
        front = mdf.zboundary(Lz)

        back.mark(boundaries, 5)
        front.mark(boundaries, 6)

    return bcs

def update_stats(t, dt, F, U, stats_df):
    c, mu = df.split(U)

    cavg = ut.integrate_func(c) / msize  # concentration average
    μavg = ut.integrate_func(mu) / msize  # chemical potential average
    Favg = ut.integrate_func(F[0]) / msize  # Helmholtz free energy

    #memUsed, memPeak = ut.memory_usage(as_string=False)

    if hlp.isHead:
        index = [0] if stats_df is None else [len(stats_df)]
        timer = time.time() - startTime
        values = np.array([[t * tc, dt * tc, cavg, μavg, Favg, timer]])#, memUsed, memPeak]])
        update = pd.DataFrame(values, columns=stats_head, index=index)

        if stats_df is None:
            return update
        else:
            return pd.concat([stats_df, update])

def output_stats(nrg_file, stats_df, report=False):
    if hlp.isHead:
        stats_df.to_csv(nrg_file, sep="\t", index=False)
        if report:
            latest = stats_df.iloc[-1]
            io.disp("├─────────────────────────────────┐")
            io.disp("│ comp c       = {:<10.6f}".format(latest["c"]))
            io.disp("│ chem pot μ   = {:<10.6f}".format(latest["μ"]))
            io.disp("│ enrg 𝓕       = {:<-10.5f}".format(latest["Helmholtz"]))
            io.disp("├─────────────────────────────────┘")

def output_data(t, F, U, fs, P1, mesh):
    io.write_fields(U, t, names, fs)

    for n, f in zip(auxnames, F):
        if n == "$\sigma$":
            if Lz is not None:
                io.write_aux(
                    f, t, n, df.TensorFunctionSpace(mesh, "DG", degree=0, shape=(6,)), fs)
            else:
                io.write_aux(
                    f, t, n, df.TensorFunctionSpace(mesh, "DG", degree=0, shape=(3,)), fs)
        else:
            io.write_aux(f, t, n, df.FunctionSpace(mesh, P1), fs)

if BDF2:
    nprev = 2
    time_bdf2 = [df.Constant(0), df.Constant(0), df.Constant(0)]
else:
    nprev = 1
    time_bdf2 = None

initStart = time.time()

io.disp("┌─ Setting up simulation.")
# Create mesh and build function space
# Initial mesh

if Lz is None:
    imesh = df.RectangleMesh.create(
        [df.Point(0, 0), df.Point(Lx, Ly)],
        [Nx, Ny],
        df.CellType.Type.triangle,
        Domain["diag"],
    )
    orgmesh = df.RectangleMesh.create(
        [df.Point(0, 0), df.Point(Lx, Ly)],
        [Nx, Ny],
        df.CellType.Type.triangle,
        Domain["diag"],
    )
    finemesh = df.RectangleMesh.create(
        [df.Point(0, 0), df.Point(Lx, Ly)],
        [int(Nx*4), int(Ny*4)],
        df.CellType.Type.quadrilateral,
    )
else:
    imesh = df.BoxMesh.create(
        [df.Point(0, 0, 0), df.Point(Lx, Ly, Lz)],
        [Nx, Ny, Nz],
        df.CellType.Type.tetrahedron,
    )
    orgmesh = df.BoxMesh.create(
        [df.Point(0, 0, 0), df.Point(Lx, Ly, Lz)],
        [Nx, Ny, Nz],
        df.CellType.Type.tetrahedron,
    )

if restartFlag:

    P1, element, ME, nmesh, U, U0, t, dt = io.read_restart(run, dt, nprev, orgmesh, Lx)

    resolution = amr.calc_resolution(emax, emin)
    imesh = df.Mesh(imesh) #amr.initial_mesh_elect(orgmesh, resolution, emax)

    U0[0].assign(U)
    P1_fine = df.FiniteElement("Lagrange", finemesh.ufl_cell(), 1)

else:

    P1 = df.FiniteElement("Lagrange", imesh.ufl_cell(), 1)
    P1_fine = df.FiniteElement("Lagrange", finemesh.ufl_cell(), 1)
    element = df.MixedElement([P1, P1])

    resolution = amr.calc_resolution(emax, emin)
    nmesh = amr.initial_mesh(imesh, resolution)

    U, U0, ME = mdf.set_functions(nmesh, Lx, element, nprev)
    
    # Create intial conditions and interpolate
    U_init = InitialConditions(degree=1)
    U.interpolate(U_init) #error here
    U0[0].assign(U)
    
bcs = []
bcs = boundaries(nmesh, ME, bcs)
# Create nonlinear problem and solver
L, *F = equation_solve(U, U0, ME, nmesh, P1, dt, BDF2, time_bdf2)
problem = slf.set_problem(L, U, ME, bcs)
solver = df.PETScSNESSolver()

msize = df.assemble(
    df.interpolate(df.Constant(1.0), df.FunctionSpace(nmesh, P1)) * df.dx
)

# Step in time

if not restartFlag:

    t = 0.0

    c_init = df.assemble(U[0] * df.dx)

    stats_df = update_stats(t, dt(0), F, U, None)

    output_stats(nrg_file, stats_df, report=True)

    output_data(t, F, U, fs, P1, nmesh)

    io.disp("├─ Saving initial state:", end=" \n")

    # Write summary record to YAML
    if hlp.isHead:
        with open("{}/record.yml".format(run), "w") as fh:
            yml = ruamel.yaml.YAML(typ=["unsafe", "pytypes"])
            yml.allow_unicode = True
            yml.default_flow_style = False
            yml.explicit_start = True
            yml.indent(mapping=2, sequence=4, offset=0)
            yml.dump(record, fh)

# iteraction counter for converged solutions
iter_s = 1

if BDF2:
    # generate second condition and redefine problem
    io.disp("├─ Step 1…")
    io.disp("├─── BDF2 secondary condition …")
    eulerStart = time.time()

    L, *F = equation_solve(U, U0, ME, nmesh, P1, dt, False, time_bdf2)
    problem = slf.set_problem(L, U, ME, bcs)

    BDF2_ini_count = 0
    converged = False

    while not converged:
        BDF2_ini_count += 1
        try:
            U0[0].assign(U)
            nits, converged = solver.solve(problem, U.vector())
        except RuntimeError:
            converged = False

        if converged:
            t += dt(0)
            iter_s += 1
            time_bdf2[1].assign(time_bdf2[0](0))
            time_bdf2[0].assign(t)
        else:
            dt.assign(dt(0) / 2)  # assign new dt to be half
            io.disp("Time step DECLINE ===> dt={}".format(dt(0)))
            U.assign(U0[0])

            if BDF2_ini_count > 10:
                io.disp("Failed 10× to calculate secondary BDF2 condition")
                io.disp("==== Terminating Simulation ====")
                hlp.mpiComm.Abort()

    # Re-define problem and form
    L, *F = equation_solve(U, U0, ME, nmesh, P1, dt, BDF2, time_bdf2)
    problem = slf.set_problem(L, U, ME, bcs)

    output_data(t, F, U, fs, P1, nmesh)

    grammar = "try" if BDF2_ini_count == 1 else "tries"
    timer = time.time() - eulerStart

    io.disp(f"├── {BDF2_ini_count} {grammar} ({nits} iters) in {timer:6.0f} s.")

stats_df = update_stats(t, dt(0), F, U, None)

output_stats(nrg_file, stats_df, report=True)
timer = time.time() - initStart
io.disp(f"└─ System Initialized in {timer:.0f} s\n")

# === Evolve the System Microstructure ===

time_to_refine = Adapt["remesh"]
time_to_refine -= 1
time_to_print = Interval["domain"]
time_to_print -= 1
amrStart = Adapt["amrStart"]

adaptTime = 0

# flag for simulation continuance
goSim = True

io.disp(f"┌─ Starting the system evolution of microstructure.")
io.disp(f"│")

flag_interval_output = 0

def apply_predictor(U, U0, mode="linear", iter_s=0, alpha=0.5):

    if mode == "linear": # or (mode == "hybrid" and iter_s < 3):
        # Formula: 2*Un - Un-1
        U.vector()[:] = 2.0 * U0[0].vector() - U0[1].vector()
        #io.disp("│ ├─ Predictor: Linear")

    elif mode == "parabolic" and iter_s >= 3: # or (mode == "hybrid" and iter_s >= 3):
        # Formula: 3*Un - 3*Un-1 + Un-2
        # Note: Requires nprev >= 3
        U.vector()[:] = 3.0 * U0[0].vector() - 3.0 * U0[1].vector() + U0[2].vector()
        #io.disp("│ ├─ Predictor: Parabolic")

    # Hybrid extrapolation: U_{n+1} = (1-α)*(2*Un - Un-1) + α*(3*Un - 3*Un-1 + Un-2)
    elif mode == "hybrid" and iter_s >= 3:
        # Formula: 3*Un - 3*Un-1 + Un-2
        # Note: Requires nprev >= 3
        U.vector()[:] = (1 - alpha) * (2.0 * U0[0].vector() - U0[1].vector()) + alpha * (3.0 * U0[0].vector() - 3.0 * U0[1].vector() + U0[2].vector())
        #io.disp("│ ├─ Predictor: Hybrid")


while goSim:
    try:
        step = dt(0)

        if time_to_refine == Adapt["remesh"]:
            leaf = "┌"
        elif time_to_refine == 1:
            leaf = "└"
        else:
            leaf = "├"

        io.disp(
            f"│  {leaf}─ Step {iter_s:<7d}| 𝑡->{t:>10f} / {T:<10f} with Δ𝑡={step:<6f}…"
        )
        t += step

        stepStart = time.time()

        for i in range(nprev - 1, 0, -1):
            U0[i].assign(U0[i - 1])

        U0[0].assign(U)

        if BDF2:
            for i in range(nprev, 0, -1):
                time_bdf2[i].assign(time_bdf2[i - 1](0))
            time_bdf2[0].assign(t)

        apply_predictor(U, U0, mode="linear", iter_s=iter_s, alpha=0.5)

        nits, converged = solver.solve(problem, U.vector())
        residual = solver.snes().norm
        iter_s += 1

        stats_df = update_stats(t, step, F, U, stats_df)

        timer = time.time() - stepStart
        io.disp(
            f"│  ├───── {timer:6.2f} s: {nits:>2d} iterations, residual={residual::<12g} "
        )

        time_to_print -= 1

        if time_to_print == 0:
            time_to_print = Interval["domain"]
            output_data(t, F, U, fs, P1, nmesh)
            output_stats(nrg_file, stats_df, report=True)

        if flag_interval_output == 1:
            flag_interval_output = 0
            io.disp(f"│ ├─ Writing time interval output")
        
            conc = U.split(deepcopy=True)[0]
            c_print = df.Function(df.FunctionSpace(finemesh, P1_fine)) 
            df.LagrangeInterpolator.interpolate(c_print,conc)
            io.write_aux(c_print, t, "c", df.FunctionSpace(finemesh, P1_fine), fs_fine)
        
            output_data(t, F, U, fs, P1, nmesh)
            output_stats(nrg_file, stats_df, report=True)

        # Iteration adaptive timestepper
        if nits == min_its:
            dt_checkpoint = abs(t%-t_interval_output)
            if dt_checkpoint > 0 and dt_checkpoint < dt(0) and dt_checkpoint < dt_max:
                dt.assign(dt_checkpoint)
                flag_interval_output = 1

        if nits < min_its:
            dt_checkpoint = abs(t%-t_interval_output)
            if dt_checkpoint  > 0 and dt_checkpoint < (dt(0)*growth) and dt_checkpoint < dt_max:
                dt.assign(dt_checkpoint)
                flag_interval_output = 1
            else:
                dt.assign(dt(0) * growth)
                if dt(0) >= dt_max:
                    dt.assign(dt_max)

        elif nits > max_its:
            dt_checkpoint = abs(t%-t_interval_output)
            if dt_checkpoint > 0 and dt_checkpoint < (dt(0)*cback) and dt_checkpoint < dt_max:
                dt.assign(dt_checkpoint)
                flag_interval_output = 1
            else:
                dt.assign(dt(0) * cback)

        if iter_s % checkpoint_frequency == 0:
            io.disp(f"│ ├─ Writing checkpoint")
            if restartFlag:
                io.output_restart(restart_run, imesh, nmesh, U, t, dt)
            else:
                io.output_restart(run, imesh, nmesh, U, t, dt)

        if iter_s > amrStart or restartFlag == True:
            time_to_refine -= 1
            if time_to_refine == 0:
                time_to_refine = Adapt["remesh"]

                df.MPI.barrier(hlp.mpiComm)
                io.disp(f"│ ")
                io.disp(f"│ ┌─      *** Starting remeshing cycle ***")

                adaptStart = time.time()

                Ur = U.copy(deepcopy=True)
                if BDF2:
                    Ur0 = U0[0].copy(deepcopy=True)

                io.disp(f"│ ├─      Current solution copied")

                nmesh = amr.refine_coarsen(
                    Lx, imesh, nmesh, P1, resolution, Ur, Adapt["tol"], Adapt["beta"]
                )

                # Define functions
                U, U0, ME = mdf.set_functions(nmesh, Lx, element, nprev)

                io.disp(f"│ ├─      Function space and solution vectors re-defined")

                df.LagrangeInterpolator.interpolate(U, Ur)
                if BDF2:
                    df.LagrangeInterpolator.interpolate(U0[0], Ur0)

                io.disp(f"│ ├─      Projection complete")

                bcs = []
                bcs = boundaries(nmesh, ME, bcs)
                L, *F = equation_solve(U, U0, ME, nmesh, P1, dt, BDF2, time_bdf2)
                problem = slf.set_problem(L, U, ME, bcs)
                solver = df.PETScSNESSolver()

                timer = time.time() - adaptStart
                io.disp(f"│ ├─      Problem redefined")
                io.disp(f"│ ├─      Refinement cycle took {timer:5.2g} s")
                io.disp(f"│ └─      *** Finished remeshing cycle ***")
                io.disp(f"│ ")
                df.MPI.barrier(hlp.mpiComm)

                adaptTime += timer

        gc.collect()
    except:
        t -= dt(0)

        #time_to_print += 1
        #flag_interval_output = 0

        io.disp("├─ Failure. {} its, 𝑡-={}".format(nits, dt(0)))

        dt.assign(dt(0) * cback)

        U.vector()[:] = U0[0].vector()
        for i in range(nprev - 1):
            U0[i].vector()[:] = U0[i + 1].vector()

        if BDF2:
            for i in range(nprev):
                time_bdf2[i].assign(time_bdf2[i + 1](0))

timer = time.time() - startTime
time_h, time_s = divmod(int(timer), int(3600))
time_m, time_s = divmod(int(time_s), int(60))

adapttime_h, adapttime_s = divmod(int(adaptTime), int(3600))
adapttime_m, adapttime_s = divmod(int(adapttime_s), int(60))

io.disp(f"│")
io.disp(
    f"├─ Total time spent remeshing {adaptTime:.2f} s or ({adapttime_h:2d}:{adapttime_m:02d}:{adapttime_s:02d})"
)
io.disp(
    f"└─ Simulation took {timer:.2f} s ({time_h:2d}:{time_m:02d}:{time_s:02d}) to reach {t:.2f}"
)
# List timings; average across processes in parallel
df.list_timings(df.TimingClear.keep, [df.TimingType.wall])
