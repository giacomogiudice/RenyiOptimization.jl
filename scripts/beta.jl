using RenyiOptimization
using OptimKit
using TensorKit
using JLD2
using ArgParse

include("auxiliary.jl")

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--beta", "-b"
            help = "inverse temperature of the simulation"
            arg_type = Float64
            required = true
        "--gamma", "-g"
            help = "interpolation parameter between XY and Ising model"
            arg_type = Float64
            default = 1.0
        "--hx"
            help = "parallel field"
            arg_type = Float64
            default = 0.0
        "--hz"
            help = "transverse field"
            arg_type = Float64
            default = 0.0
        "--Z2"
            help = "enforce ℤ₂ symmetry"
            action = :store_true
        "--maxiter", "-n"
            help = "maximum number of iterations per bond dimensions"
            arg_type = Int
            default = 10000
        "--tol", "-t"
            help = "tolerance on gradient magnitude"
            arg_type = Float64
            default = 1e-6
        "--verbosity", "-v"
            help = "verbosity level of output"
            arg_type = Int
            default = 2
        "--bonddims", "-D"
            help = "bond dimensions for each simulation"
            arg_type = Int
            required = true
            nargs = '*'
        "--preconditioner"
            help = "use preconditioner"
            action = :store_true
        "--dir"
            help = "location of output file"
            arg_type = String
            required = true
    end

    parse_args(s; as_symbols = true)
end

function main()
    # Parse input parameters
    pars = parse_commandline()
    @debug "Parsed args: $pars"

    # User-defined parameters go here
    β = pars[:beta]
    γ = pars[:gamma]
    hx = pars[:hx]
    hz = pars[:hz]
    group = pars[:Z2] ? ℤ₂ : nothing
    maxiter = pars[:maxiter]
    tol = pars[:tol]
    verbosity = pars[:verbosity]
    Ds = pars[:bonddims]
    preconditioner = pars[:preconditioner]
    location = pars[:dir]
    
    @debug "Main routine started"

    # Hardcoded parameters go here
    type = Float64
    linesearch = HagerZhangLineSearch()
    alg = LBFGS(20; maxiter = maxiter, verbosity = verbosity, gradtol = tol, linesearch = linesearch)
    solvertol = 1e-12

    # The ol' Pauli matrices
    X, Y, Z = map(σ -> TensorMap(σ, ℂ^2 ← ℂ^2), ([0 1; 1 0], [0 -im; im 0], [1 0; 0 -1]))

    # Define the 2-site contribution to the XY model hamiltonian
    𝟙 = one(Z)
    h = -(1+γ)/2*(X ⊗ X) - (1-γ)/2*real(Y ⊗ Y) - hz/2*(𝟙 ⊗ Z + Z ⊗ 𝟙) - hx/2*(𝟙 ⊗ X + X ⊗ 𝟙)

    D₀ = first(Ds)
    if group == ℤ₂
        hx ≈ 0 || @warn "field hₓ is not zero"
        indmap = [ℤ₂(0),ℤ₂(1)]
        Z = restrict(Z, indmap)
        X = zero(Z)
        Y = zero(Z)

        h = restrict(h, indmap)
        Dsect = Int(D₀/2)
        virt = ℂ[ℤ₂](0 => Dsect, 1 => Dsect)
    else
        virt = ℂ^D₀
    end
    phys = space(Z, 1)
    A = TensorMap(randisometry, type, virt ⊗ phys ⊗ phys ← virt)

    # Loop over different bond dimensions
    for D in Ds
        @info "Running D = $(D), β = $(β)..."
        if D == D₀
            A = TensorMap(randisometry, type, virt ⊗ phys ⊗ phys ← virt)
        else
            A = increasebond(A, h; β = β, trunc = truncdim(D - D₀)) 
        end
        # Run the optimization
        A, output, ρL, ρR, gradhistory  = betaoptimize(β, h, A, alg; tol = solvertol, preconditioner = preconditioner)

        # Compute observables
        observables =
        (
            h = output.ε,
            X = expectationvalue(X, A, ρL, ρR),
            Z = expectationvalue(Z, A, ρL, ρR),
            XX = two_point_correlations(X, X, A, [1], ρL, ρR) |> first,
            ZZ = two_point_correlations(Z, Z, A, [1], ρL, ρR) |> first
        )
        
        # Save results
        filename = "$(location)/XY__γ=$(γ)_hx=$(hx)_hz=$(hz)_β=$(β)_group=$(group)_D=$(D).jld2"
        @save filename β A ρL ρR observables type linesearch alg solvertol group D output gradhistory preconditioner
    end
    return Cint(0)
end

main()
