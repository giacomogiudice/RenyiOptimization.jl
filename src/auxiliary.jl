"""
    udot(L::AbstractTensorMap{S,N₁,N₂}, R::AbstractTensorMap{S,N₂,N₁})

Compute the unconjugated dot product, or equivalently tr(L*R). This operation is mapped to the unicode symbol ⊙.
"""
@inline udot(L::AbstractTensorMap{S,N₁,N₂}, R::AbstractTensorMap{S,N₂,N₁}) where {S,N₁,N₂} = tr(L*R)
@inline ⊙(L::AbstractTensorMap{S,N₁,N₂}, R::AbstractTensorMap{S,N₂,N₁}) where {S,N₁,N₂} = udot(L, R)

"""
    rinv(t::AbstractTensorMap, ϵ::Real=eps(real(eltype(t))))

Return the Tikhonov-regularized inverse of a tensor, using the SVD decomposition. Each singular value is inverted as 

Σᵢ⁻¹ = Σᵢ/(Σᵢ² + ϵ²)

which decreases the condition number of ill-posed problems.
"""
function rinv(t::AbstractTensorMap, ϵ::Real=eps(real(eltype(t))))
    U, Σ, V = tsvd(t)
    Σ⁻ = (Σ^2 + ϵ^2*one(Σ))\Σ
    return U*Σ⁻*V
end
