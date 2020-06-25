function singlefixedpoints(AL::PurificationTensor; kwargs...)
    ρR = TensorMap(randn, eltype(AL), rightvirtual(AL) ← rightvirtual(AL))
    ρL = one(ρR)
    return singlefixedpoints(AL, ρL, ρR; kwargs...)
end

function singlefixedpoints(AL::PurificationTensor{S}, ρL::BondTensor{S}, ρR::BondTensor{S}; kwargs...) where S
    _, ρRs, ζs, info = schursolve(ρ -> righttransfer(ρ, AL), ρR, 1, :LM, Arnoldi(; kwargs...))
    info.converged == 0 && @warn "No convergence in singlefixedpoints \n $info"
    
    ρR = ρRs[1]
    rmul!(ρR, 1/tr(ρR))
    ζ::real(eltype(AL)) = abs(ζs[1])

    ∇ζ = 2*AL*ρR

    return ρL, ρR, ζ, ∇ζ
end

function doublefixedpoints(AL::PurificationTensor; kwargs...)
    ΣL = TensorMap(rand, eltype(AL), leftvirtual(AL)^2 ← leftvirtual(AL)^2)
    ΣR = TensorMap(rand, eltype(AL), rightvirtual(AL)^2 ← rightvirtual(AL)^2)
    return doublefixedpoints(AL, ΣL, ΣR; kwargs...)
end

function doublefixedpoints(A::PurificationTensor{S}, ΣL::DoubleBoundary{S}, ΣR::DoubleBoundary{S}; kwargs...) where S
    _, ΣLs, ηLs, infoL = schursolve(Σ -> leftdoubletransfer(Σ, A), ΣL, 1, :LM, Arnoldi(; kwargs...))
    _, ΣRs, ηRs, infoR = schursolve(Σ -> rightdoubletransfer(Σ, A), ΣR, 1, :LM, Arnoldi(; kwargs...))
    (infoL.converged == 0 || infoR.converged == 0) && @warn "No convergence in doublefixedpoints \n $infoL, $infoR"
    
    ΣL = ΣLs[1]
    ΣR = ΣRs[1]
    rmul!(ΣR, 1/(ΣL ⊙ ΣR))
    η::real(eltype(A)) = abs(ηLs[1] + ηRs[1])/2

    @tensor ∇η[-1 -2 -3; -4] := 2 * ΣL[-1 1 5; 3] * conj(A[1 2 4; 6]) * A[5 -2 4; 7] * A[3 2 -3; 8] * ΣR[7 8 -4; 6]
    
    return ΣL, ΣR, η, ∇η
end

function energyfixedpoints(A::PurificationTensor{S}, H::TwoSiteOperator{S}, ρL::BondTensor{S}, ρR::BondTensor{S}; kwargs...) where S
    return energyfixedpoints(A, H, ρL, ρR, ρL, ρR; kwargs...)
end

function energyfixedpoints(A::PurificationTensor{S}, H::TwoSiteOperator{S}, ρL::BondTensor{S}, ρR::BondTensor{S}, HL::BondTensor{S}, HR::BondTensor{S}; kwargs...) where S
    @tensor begin
        hL[-1; -2] := ρL[5; 1] * A[1 3 7; 2] * A[2 4 10; -2] * H[6 9; 3 4] * conj(A[5 6 7; 8]) * conj(A[8 9 10; -1])
        hR[-1; -2] := ρR[1; 5] * A[-1 4 10; 2] * A[2 3 7; 1] * H[9 6; 4 3] * conj(A[-2 9 10; 8]) * conj(A[8 6 7; 5])
    end
    εL = hL ⊙ ρR
    εR = ρL ⊙ hR
    ε::real(eltype(A)) = real(εL + εR)/2

    HL, infoL = linsolve(hL - εL*ρL, HL; kwargs...) do X
        X - lefttransfer(X, A) + (X ⊙ ρR)*ρL
    end
    HR, infoR = linsolve(hR - ρR*εR, HR; kwargs...) do X
        X - righttransfer(X, A) + ρR*(ρL ⊙ X)
    end
    (infoL.converged == 0 || infoR.converged == 0) && @warn "No convergence in energyfixedpoints \n $infoL, $infoR"

    @tensor begin
        T[-1 -2 -3 -4 -5; -6] := ρL[-1; 1] * A[1 4 -4; 3] * H[-2 -3; 4 5] * A[3 5 -5; 2] * ρR[2; -6]
        ∇ε[-1 -2 -3; -4] := conj(A[1 2 3; -1]) * T[1 2 -2 3 -3; -4] + T[-1 -2 2 -3 3; 1] * conj(A[-4 2 3; 1]) +
                            HL[-1; 1] * A[1 -2 -3; 2] * ρR[2; -4] + ρL[-1; 1] * A[1 -2 -3; 2] * HR[2; -4]
    end

    return HL, HR, ε, ∇ε
end

# function doublefixedpoints2(A::PurificationTensor{S}, ΣL::AbstractTensorMap{S,2,2}, ΣR::AbstractTensorMap{S,2,2}; kwargs...) where S

#     @tensor T[-1 -2 -3; -4 -5 -6] := conj(A[-1 1; -3 -5]) * A[-2 1; -4 -6]
#     _, ΣLs, ηLs, infoL = schursolve(ΣL, 1, :LM, Arnoldi(; kwargs...)) do Σ
#         @tensor Σ′[-1 -2; -3 -4] := Σ[1 4; 2 5] * T[1 2 3; 6 -1 -3] * T[4 5 6; 3 -2 -4]
#     end
#     _, ΣRs, ηRs, infoR = schursolve(ΣR, 1, :LM, Arnoldi(; kwargs...)) do Σ
#         @tensor Σ′[-1 -2; -3 -4] := Σ[1 4; 2 5] * T[-3 -1 3; 6 2 1] * T[-4 -2 6; 3 5 4]
#     end

#     (infoL.converged == 0 || infoR.converged == 0) && @warn "No convergence in doublefixedpoints \n $infoL, $infoR"
#     ΣL = ΣLs[1]
#     ΣR = ΣRs[1]
#     rmul!(ΣR, 1/(ΣL ⊙ ΣR))
#     η = abs(ηLs[1] + ηRs[1])/2

#     @tensor ∇η[-1 -2; -3 -4] := 2 * ΣL[-1 1; 5 3] * conj(A[1 2; 4 6]) * A[5 -2; 4 7] * A[3 2; -3 8] * ΣR[7 8; -4 6]
#     return ΣL, ΣR, η, ∇η
# end
