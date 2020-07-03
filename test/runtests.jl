using RenyiOptimization, TensorKit
using Test, Random, LinearAlgebra

using RenyiOptimization: initialize, fg, project!, inner, retract, transport
import OptimKit

# Set seed for reproducibility
Random.seed!(42)

phys_spaces = (ℂ^2, ℂ^3, ℂ^2)
virt_spaces = (ℂ^2, ℂ^5, ℂ^4)
anc_spaces = (ℂ^2, ℂ^3, ℂ^3)

@testset "Fixed points tests for type $(T)" for T in (Float32, Float64, ComplexF32, ComplexF64)
    tol = eps(real(T))
    testtol = 100*tol

    @testset "Physical $(phys), virtual $(virt), ancillar $(anc)" for (phys, virt, anc) in zip(phys_spaces, virt_spaces, anc_spaces)
        AL = TensorMap(randisometry, T, virt ⊗ phys ⊗ anc ← virt)

        O = TensorMap(randn, T, phys ← phys)
        O = O + O'
        𝟙 = one(O)
        H = convert(T, 1/2)*(O ⊗ 𝟙 + 𝟙 ⊗ O)

        @testset "Basics" begin
            @test leftvirtual(AL) == rightvirtual(AL) == virt
            @test physical(AL) == phys
            @test ancillar(AL) == anc
            @test isleftgauged(AL)
        end

        @testset "Single fixed points" begin
            ρL, ρR, ζ, ∇ζ = @inferred singlefixedpoints(AL; tol = tol)

            @test ζ ≈ 1 atol = testtol
            @test ρL ≈ one(ρL) atol = testtol
            @test ρL ⊙ ρR ≈ 1 atol = testtol
            @test righttransfer(ρR, AL) ≈ ρR atol = testtol
            @test (∇ζ ⋅ AL)/2 ≈ 1 atol = testtol
        end

        @testset "Double fixed points" begin
            ΣL, ΣR, η, ∇η = @inferred doublefixedpoints(AL; tol = tol)

            @test ΣL ⊙ ΣR ≈ 1 atol = testtol
            @test norm(leftdoubletransfer(ΣL, AL) - η*ΣL, Inf) ≈ 0 atol = testtol
            @test norm(rightdoubletransfer(ΣR, AL) - η*ΣR, Inf) ≈ 0 atol = testtol
            @test (∇η ⋅ AL)/2 ≈ η atol = testtol
        end

        @testset "Energy fixed points" begin
            ρL, ρR, _, _ = singlefixedpoints(AL; tol = tol)
            HL, HR, ε, ∇ε = @inferred energyfixedpoints(AL, H, ρL, ρR; tol = tol)

            @test HL ⊙ ρR ≈ 0 atol = testtol
            @test ρL ⊙ HR ≈ 0 atol = testtol
            @test ε ≈ expectationvalue(O, AL) atol = testtol
            @test (∇ε ⋅ AL)/2 ≈ ε atol = testtol
        end
    end 
end

# Finite differences tests fail miserably with single-precision
@testset "Manifold tests for type $(T)" for T in (Float64, ComplexF64)
    tol = eps(real(T))
    testtol = 100*tol

    @testset "Physical $(phys), virtual $(virt), ancillar $(anc)" for (phys, virt, anc) in zip(phys_spaces, virt_spaces, anc_spaces)
        AL = TensorMap(randisometry, T, virt ⊗ phys ⊗ anc ← virt)

        O = TensorMap(randn, T, phys ← phys)
        O = O + O'
        𝟙 = one(O)
        H = convert(T, 1/2)*(O ⊗ 𝟙 + 𝟙 ⊗ O)

        x = initialize(AL, H; tol = tol)

        ξ = @inferred project!(TensorMap(randn, T, codomain(AL) ← domain(AL)), x)
        Δ₁ = project!(TensorMap(randn, T, codomain(AL) ← domain(AL)), x)
        Δ₂ = project!(TensorMap(randn, T, codomain(AL) ← domain(AL)), x)

        @testset "Consistency" begin
            @test inner(x, ξ, project!(ξ[], x)) ≈ @inferred inner(x, ξ, ξ)
            
            x′, ξ′ = @inferred retract(x, ξ, 0; tol = tol)
            @test norm(first(x) - first(x′)) ≈ 0 atol = testtol
            @test inner(x′, ξ′, ξ′) ≈ inner(x, ξ, ξ) rtol = testtol

            x′, _ = retract(x, ξ, 0; tol = tol)
            Δ′ = @inferred transport(Δ₁, x, ξ, 0, x′)
            @test norm(first(x) - first(x′)) ≈ 0 atol = testtol
            @test inner(x′, Δ′, Δ′) ≈ inner(x, Δ₁, Δ₁) rtol = testtol
        end

        @testset "Isometric transport" begin
            αs = 10.0.^(-5:0)
            for α in αs
                x′, _ = retract(x, ξ, α; tol = tol)
                @test inner(x, Δ₁, Δ₂) ≈ inner(x′, transport(Δ₁, x, ξ, α, x′), transport(Δ₂, x, ξ, α, x′)) rtol = testtol
            end
        end

        @testset "Finite differences" begin
            x = @inferred initialize(AL, H; tol = tol)
            αs = 10.0.^(-10:-4)
            αs, fs, dfs1, dfs2 = @inferred OptimKit.optimtest(fg, x; alpha = αs, retract = (x, ξ, α) -> retract(x, ξ, α; tol = tol), inner = inner)
            @test norm(dfs1 - dfs2, Inf) ≈ 0 atol = 1e-5    # Not very robust
        end
    end
end

@testset "Optimization tests for type $(T)" for T in (Float64, ComplexF64)
    tol = √eps(real(T))
    testtol = tol
    virt = phys = anc = ℂ^2
    O = TensorMap(randn, T, phys ← phys)
    O = O + O'
    H = TensorMap(zeros, T, phys^2 ← phys^2)
    AL = TensorMap(randisometry, T, virt ⊗ phys ⊗ anc ← virt)
    alg = OptimKit.LBFGS(20; maxiter = 1000, verbosity = 0, gradtol = tol)
    solvertol = 1e-3*tol
    @testset "Optimization without preconditioner" begin
        AL, output, ρL, ρR, _ = renyioptimize(1, H, AL, alg; preconditioner = false, tol = solvertol)
        @test output.f ≈ log(0.5) atol = testtol
        @test output.η ≈ 0.5 atol = testtol
        @test output.ε ≈ 0 atol = testtol
        @test norm(two_point_correlations(O, O, AL, 1:10, ρL, ρR) .- expectationvalue(O, AL, ρL, ρR)^2, Inf) ≈ 0 atol = testtol
    end

    @testset "Optimization with preconditioner" begin
        AL, output, ρL, ρR, _ = renyioptimize(1, H, AL, alg; preconditioner = true, tol = solvertol)
        @test output.f ≈ log(0.5) atol = testtol
        @test output.η ≈ 0.5 atol = testtol
        @test output.ε ≈ 0 atol = testtol
        @test norm(two_point_correlations(O, O, AL, 1:10, ρL, ρR) .- expectationvalue(O, AL, ρL, ρR)^2, Inf) ≈ 0 atol = testtol
    end

end
