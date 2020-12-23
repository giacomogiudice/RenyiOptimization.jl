function betaoptimize(β::Real, H::TwoSiteOperator{S}, AL::PurificationTensor{S}, alg::OptimKit.OptimizationAlgorithm; preconditioner = false, kwargs...) where S
    # Construct initial object
    x = initialize(AL, H; kwargs...)
    # Fire up the optimization
    fg = x -> betafg(x, β)
    x, fx, _, gradhistory = optimize(fg, x, alg;
                                    retract = (x, Δ, α) -> retract(x, Δ, α; kwargs...),
                                    inner = inner,
                                    transport! = transport!,
                                    precondition = preconditioner == true ? precondition : OptimKit._precondition,
                                    isometrictransport = true)

    # Unpack x object
    AL, H, ρL, ρR, ζ, ∇ζ, ΣL, ΣR, η, ∇η, HL, HR, ε, ∇ε = x
    return AL, (f = fx, η = η, ε =  ε), ρL, ρR, gradhistory
end

function lambdaoptimize(λ::Real, ε₀::Real, H::TwoSiteOperator{S}, AL::PurificationTensor{S}, alg::OptimKit.OptimizationAlgorithm; preconditioner = false, kwargs...) where S
    # Construct initial object
    x = initialize(AL, H; kwargs...)
    fg = x -> lambdafg(x, λ, ε₀)
    # Fire up the optimization
    x, fx, _, gradhistory = optimize(fg, x, alg;
                                    retract = (x, Δ, α) -> retract(x, Δ, α; kwargs...),
                                    inner = inner,
                                    transport! = transport!,
                                    precondition = preconditioner == true ? precondition : OptimKit._precondition,
                                    isometrictransport = true)

    # Unpack x object
    AL, H, ρL, ρR, ζ, ∇ζ, ΣL, ΣR, η, ∇η, HL, HR, ε, ∇ε = x
    return AL, (f = fx, η = η, ε = ε), ρL, ρR, gradhistory
end

function initialize(AL::PurificationTensor{S}, H::TwoSiteOperator{S}; kwargs...) where S
    # Compute initial fixed points
    ρL, ρR, ζ, ∇ζ = singlefixedpoints(AL; kwargs...)
    ΣL, ΣR, η, ∇η = doublefixedpoints(AL; kwargs...)
    HL, HR, ε, ∇ε = energyfixedpoints(AL, H, ρL, ρR; kwargs...)

    # Pack into x object
    x = (AL, H, ρL, ρR, ζ, ∇ζ, ΣL, ΣR, η, ∇η, HL, HR, ε, ∇ε)
    return x
end

function betafg(x, β)
    AL, H, ρL, ρR, ζ, ∇ζ, ΣL, ΣR, η, ∇η, HL, HR, ε, ∇ε = x

    # f = ε + log(η)            
    # ∇f = 2*(∇ε + 1/η*∇η)      
    f = β*ε + log(η)
    ∇f = 2*(β*∇ε + 1/η*∇η)

    Δ = project!(∇f, x)
    return f, Δ
end

function lambdafg(x, λ, ε₀)
    AL, H, ρL, ρR, ζ, ∇ζ, ΣL, ΣR, η, ∇η, HL, HR, ε, ∇ε = x

    f = 1/2*λ^2*(ε - ε₀)^2 + log(η)
    ∇f = 2*(λ^2*(ε - ε₀)*∇ε + 1/η*∇η)

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

function precondition(x, Δ)
    AL, H, ρL, ρR, ζ, ∇ζ, ΣL, ΣR, η, ∇η, HL, HR, ε, ∇ε = x

    ϵ = max(norm(Δ), eps(real(eltype(Δ[]))))
    ρ⁻¹ = rinv(ρR, ϵ)
    return Grassmann.project!(Δ[]*ρ⁻¹, AL)
end
