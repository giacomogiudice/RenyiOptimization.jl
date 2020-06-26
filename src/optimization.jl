function renyioptimize(β::Real, H::TwoSiteOperator{S}, A::PurificationTensor{S}, alg::OptimKit.OptimizationAlgorithm; expand = nothing, kwargs...) where S
    # Construct initial object
    x = initialize(AL, β*H; trunc = expand, kwargs...)
    # Fire up the optimization
    x, fx, _, gradhistory = optimize(fg, x, alg;
                                    retract = (x, Δ, α) -> retract(x, Δ, α; kwargs...),
                                    inner = inner,
                                    transport! = transport!,
                                    isometrictransport = true)

    # Unpack x object
    AL, H, ρL, ρR, ζ, ∇ζ, ΣL, ΣR, η, ∇η, HL, HR, ε, ∇ε = x
    return AL, (f = fx, η = η, ε = ε), ρL, ρR, gradhistory
end

function initialize(AL::PurificationTensor{S}, H::TwoSiteOperator{S}; expand = nothing, kwargs...) where S
    # Compute initial fixed points
    ρL, ρR, ζ, ∇ζ = singlefixedpoints(AL; kwargs...)
    ΣL, ΣR, η, ∇η = doublefixedpoints(AL; kwargs...)
    HL, HR, ε, ∇ε = energyfixedpoints(AL, H, ρL, ρR; kwargs...)

    # Pack into x object
    x = (AL, H, ρL, ρR, ζ, ∇ζ, ΣL, ΣR, η, ∇η, HL, HR, ε, ∇ε)

    if !isnothing(expand)
        _, Δ = fg(x)
        # Δ = TensorMap(rand, eltype(AL), codomain(AL) ← domain(AL))

        E, _, _ = tsvd!(Δ[]; trunc = expand)
        V = domain(E)
        AL = catdomain(AL, E)
        AL = permute(AL, (1,), (2,3,4))
        E = TensorMap(zeros, eltype(AL), V ← domain(AL))
        AL = permute(catcodomain(AL, E), (1,2,3), (4,))

        x = initialize(AL, H)
    end
    return x
end

function fg(x)
    AL, H, ρL, ρR, ζ, ∇ζ, ΣL, ΣR, η, ∇η, HL, HR, ε, ∇ε = x

    f = ε + log(η)            # f = β*ε + log(η)
    ∇f = 2*(∇ε + 1/η*∇η)      # ∇f = 2*(β*∇ε + 1/η*∇η)

    Δ = project!(∇f, x)
    return f, Δ
end

project!(g, x) = Grassmann.project!(g, first(x))

inner(x, Δ₁, Δ₂) = Grassmann.inner(first(x), Δ₁, Δ₂)

# function inner(x, Δ₁, Δ₂)
#     AL, H, ρL, ρR, ζ, ∇ζ, ΣL, ΣR, η, ∇η, HL, HR, ε, ∇ε = x
#     return real(tr(Δ₂[]'*Δ₁[]*ρR))
# end

function retract(x, Δ, α; kwargs...)
    AL, H, ρL, ρR, ζ, ∇ζ, ΣL, ΣR, η, ∇η, HL, HR, ε, ∇ε = x

    AL′, Δ′ = Grassmann.retract(AL, Δ, α)

    # Recompute all accompaning objects
    ρL, ρR, ζ, ∇ζ = singlefixedpoints(AL′, ρL, ρR; kwargs...)
    ΣL, ΣR, η, ∇η = doublefixedpoints(AL′, ΣL, ΣR; kwargs...)
    HL, HR, ε, ∇ε = energyfixedpoints(AL′, H, ρL, ρR, HL, HR; kwargs...)

    # Pack into x object
    x′ = (AL′, H, ρL, ρR, ζ, ∇ζ, ΣL, ΣR, η, ∇η, HL, HR, ε, ∇ε)
    return x′, Δ′
end

transport(Θ, x, Δ, α, x′) = Grassmann.transport(Θ, first(x), Δ, α, first(x′))
transport!(Θ, x, Δ, α, x′) = Grassmann.transport!(Θ, first(x), Δ, α, first(x′))

