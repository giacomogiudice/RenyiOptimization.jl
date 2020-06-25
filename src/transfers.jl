lefttransfer(v::BondTensor{S}, A::PurificationTensor{S}, A′::PurificationTensor{S} = A) where S =
    @tensor v′[-1; -2] := v[1; 2] * conj(A′[1 3 4; -1]) * A[2 3 4; -2]

righttransfer(v::BondTensor{S}, A::PurificationTensor{S}, A′::PurificationTensor{S} = A) where S =
    @tensor v′[-1; -2] := A[-1 3 4; 1] * conj(A′[-2 3 4; 2]) * v[1; 2]

lefttransfer(v::BondTensor{S}, A::PurificationTensor{S}, O::LocalOperator{S}, A′::PurificationTensor{S} = A) where S =
    @tensor v′[-1; -2] := v[1; 2] * conj(A′[1 3 5; -1]) * O[3; 4] * A[2 4 5; -2]

righttransfer(v::BondTensor{S}, A::PurificationTensor{S}, O::LocalOperator{S}, A′::PurificationTensor{S} = A) where S =
    @tensor v′[-1; -2] := A[-1 3 5; 1] * O[4; 3] * conj(A′[-2 4 5; 2]) * v[1; 2]

leftdoubletransfer(v::DoubleBoundary{S}, A::PurificationTensor{S}, A′::PurificationTensor{S} = A) where S =
    @tensor v′[-1 -2; -3 -4] := v[1 5; 3 7] * conj(A[1 2 8; -1]) * conj(A[5 6 4; -2]) * A[3 2 4; -3] * A[7 6 8; -4]

rightdoubletransfer(v::DoubleBoundary{S}, A::PurificationTensor{S}, A′::PurificationTensor{S} = A) where S =
    @tensor v′[-1 -2; -3 -4] := v[1 5; 3 7] * conj(A[-3 2 4; 3]) * conj(A[-4 6 8; 7]) * A[-1 2 8; 1] * A[-2 6 4; 5]
