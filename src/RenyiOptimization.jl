module RenyiOptimization

export LocalOperator, TwoSiteOperator, PurificationTensor, BondTensor, DoubleBoundary
export leftvirtual, rightvirtual, physical, ancillar, isleftgauged
export lefttransfer, righttransfer, leftdoubletransfer, rightdoubletransfer
export singlefixedpoints, doublefixedpoints, energyfixedpoints
export expectationvalue, two_point_correlations
export renyioptimize
export udot, ⊙, rinv, paulimatrices

using KrylovKit, OptimKit
using TensorKit, TensorKitManifolds
import TensorKitManifolds.Grassmann

# Typedefs of the different tensors used
const LocalOperator{S} = AbstractTensorMap{S,1,1}
const TwoSiteOperator{S} = AbstractTensorMap{S,2,2}
const PurificationTensor{S} = AbstractTensorMap{S,3,1}
const BondTensor{S} = AbstractTensorMap{S,1,1}
const DoubleBoundary{S} = AbstractTensorMap{S,2,2}


# Define here what the different indices represent
leftvirtual(A::PurificationTensor) = codomain(A, 1)
rightvirtual(A::PurificationTensor) = domain(A, 1)
physical(A::PurificationTensor) = codomain(A, 2)
ancillar(A::PurificationTensor) = codomain(A, 3)

leftvirtual(C::BondTensor) = codomain(C, 1)
rightvirtual(C::BondTensor) = domain(C, 1)

# Function to check proper gauge condition
function isleftgauged(A::PurificationTensor; kwargs...)
    type = storagetype(A)
    leftid = id(type, leftvirtual(A))
    rightid = id(type, rightvirtual(A))
    isapprox(lefttransfer(leftid, A), rightid; kwargs...)
end


include("transfers.jl")
include("fixedpoints.jl")
include("expectationvalues.jl")
include("optimization.jl")
include("auxiliary.jl")

end
