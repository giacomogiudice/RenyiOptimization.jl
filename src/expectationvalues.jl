function expectationvalue(O::LocalOperator{S}, A::PurificationTensor{S}) where S
    ρL, ρR, ζ, _ = singlefixedpoints(A)
    return expectationvalue(O, A/√ζ, ρL, ρR)
end

expectationvalue(O::LocalOperator{S}, A::PurificationTensor{S}, ρL::BondTensor{S}, ρR::BondTensor{S}) where S = lefttransfer(ρL, A, O, A) ⊙ ρR

function two_point_correlations(O₁::LocalOperator{S}, O₂::LocalOperator{S}, A::PurificationTensor{S}, dist::AbstractVector{Int}, ρL::BondTensor{S}, ρR::BondTensor{S}) where S
    σL = lefttransfer(ρL, A, O₁, A)
    vals = Vector{eltype(A)}()
    for n in 1:maximum(dist)
        if n in dist
            σR = righttransfer(ρR, A, O₂, A)
            vals = [vals; σL ⊙ σR]
        end
        σL = lefttransfer(σL, A, A)
    end
    return vals
end
