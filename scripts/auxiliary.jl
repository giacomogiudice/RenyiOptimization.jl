function increasebond(AL, H; β = nothing, λ = nothing, ε₀ = nothing, trunc = TensorKit.notrunc())
    x = RenyiOptimization.initialize(AL, H)
    # Compute the tangent vector corresponding to the gradient, and keep largest singular values
    if !isnothing(β) && isnothing(λ) && isnothing(ε₀)
        _, Δ = RenyiOptimization.betafg(x, β)
    elseif isnothing(β) && !isnothing(λ) && !isnothing(ε₀)
        _, Δ = RenyiOptimization.lambdafg(x, λ, ε₀)
    else
        error("Either β or (λ and ε₀) must be provided, but not both")
    end
    U, Σ, V = tsvd!(Δ[]; trunc = trunc)
    Ξ = U
    # Add the direction by blocks to the state
    V = domain(Ξ)
    AL = catdomain(AL, Ξ)
    AL = permute(AL, (1,), (2,3,4))
    Ξ = TensorMap(zeros, eltype(AL), V ← domain(AL))
    AL = permute(catcodomain(AL, Ξ), (1,2,3), (4,))
    return AL
end

restrict(t::AbstractTensorMap{S,N₁,N₂}, indmap::AbstractVector{G}) where {S<:IndexSpace,N₁,N₂,G<:Sector} =
    restrict(t, ntuple(_ -> indmap, N₁), ntuple(_ -> indmap, N₂))

function restrict(t::AbstractTensorMap{S,N₁,N₂}, codommaps::NTuple{N₁,<:AbstractVector{G}},
                    dommaps::NTuple{N₂,<:AbstractVector{G}}) where {S<:IndexSpace,N₁,N₂,G<:Sector}
    # Currently works only for Abelian symmetries, where two charges fuse to a common charge
    FusionStyle(G) == Abelian() || throw(NotImplemented())
    all(i -> dim(codomain(t, i)) == length(codommaps[i]), 1:N₁) ||
        throw(ArgumentError("Dimension mismatch in codomain indices"))
    all(i -> dim(domain(t, i)) == length(dommaps[i]), 1:N₂) ||
        throw(ArgumentError("Dimension mismatch in domain indices"))

    # Build domain and codomain of the output tensor
    codom = mapreduce(⊗, 1:N₁) do i
        degs = Dict([(s, count(x -> x == s, codommaps[i])) for s in unique(codommaps[i])])
        V = ℂ[G]((s => d for (s,d) in degs))
    end
    dom = mapreduce(⊗, 1:N₂) do i
        degs = Dict([(s, count(x -> x == s, dommaps[i])) for s in unique(dommaps[i])])
        V = ℂ[G]((s => d for (s,d) in degs))
    end

    # Build dictionary associated to the blocks of the output tensor
    data = Dict{G,Array{eltype(t),2}}()
    for s in blocksectors(codom ← dom)
        # Iterate over all indices of domain and codomain separately, finding the ones
        # which fuse to the desired sector
        codomind = Iterators.filter(Iterators.product(ntuple(i -> 1:length(codommaps[i]), N₁)...)) do tupind
            length(tupind) == 1 && return codommaps[1][tupind[1]] == s
            first(⊗((codommaps[i][s] for (i, s) in enumerate(tupind))...)) == s
        end
        domind = Iterators.filter(Iterators.product(ntuple(i -> 1:length(dommaps[i]), N₂)...)) do tupind
            length(tupind) == 1 && return dommaps[1][tupind[1]] == s
            first(⊗((dommaps[i][s] for (i, s) in enumerate(tupind))...)) == s
        end
        # Now iterate over all the indices in domain and codomain, and push into an array
        data[s] = [t[i...,j...] for (i,j) in Iterators.product(collect(codomind), collect(domind))]
    end

    # Check norm difference as minimal check
    mapreduce(d -> norm(d)^2, +, values(data)) ≈ norm(t)^2 ||
        @warn "Norm of restricted tensor differs from input"

    return TensorMap(data, codom ← dom)
end

