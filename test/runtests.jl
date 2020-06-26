using RenyiOptimization, TensorKit
using Test, Random, LinearAlgebra

using RenyiOptimization: initialize, fg, project!, inner, retract, transport
import OptimKit

# Set seed for reproducibility
Random.seed!(42)

@testset "Fixed points tests for type $(T)" for T in (Float32, Float64, ComplexF32, ComplexF64)
    tol = 10*eps(real(T))
    testtol = 2*tol

    phys_spaces = (ℂ^2, ℂ^4, ℂ^3)
    virt_spaces = (ℂ^2, ℂ^10, ℂ^4)
    anc_spaces = (ℂ^2, ℂ^4, ℂ^5)

    @testset "Physical $(phys), virtual $(virt), ancillar $(anc)" for (phys, virt, anc) in zip(phys_spaces, virt_spaces, anc_spaces)
        AL = TensorMap(randisometry, T, virt ⊗ phys ⊗ anc ← virt)

        O = TensorMap(randn, T, phys ← phys)
        O = (O + O')/2
        𝟙 = one(O)
        H = (O ⊗ 𝟙 + 𝟙 ⊗ O)/2

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

            @test ΣL ⊙ ΣR ≈ 1 atol = atol
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

@testset "Manifold tests for type $(T)" for T in (Float32, Float64, ComplexF32, ComplexF64)
    tol = 10*eps(real(T))
    testtol = 10*tol

    phys_spaces = (ℂ^2, ℂ^4, ℂ^3)
    virt_spaces = (ℂ^2, ℂ^10, ℂ^4)
    anc_spaces = (ℂ^2, ℂ^4, ℂ^5)

    @testset "Physical $(phys), virtual $(virt), ancillar $(anc)" for (phys, virt, anc) in zip(phys_spaces, virt_spaces, anc_spaces)
        AL = TensorMap(randisometry, T, virt ⊗ phys ⊗ anc ← virt)

        O = TensorMap(randn, T, phys ← phys)
        O = (O + O')/2
        𝟙 = one(O)
        H = (O ⊗ 𝟙 + 𝟙 ⊗ O)/2

        x = initialize(AL, H; tol = tol)

        ξ = @inferred project!(TensorMap(randn, T, codomain(AL) ← domain(AL)), x)
        Δ₁ = project!(TensorMap(randn, T, codomain(AL) ← domain(AL)), x)
        Δ₂ = project!(TensorMap(randn, T, codomain(AL) ← domain(AL)), x)

        @testset "Consistency" begin
            @test inner(x, ξ, project!(ξ[], x)) ≈ @inferred inner(x, ξ, ξ)
            
            x′, ξ′ = @inferred retract(x, ξ, 0)
            @test norm(first(x) - first(x′)) ≈ 0 atol = testtol
            @test inner(x′, ξ′, ξ′) ≈ inner(x, ξ, ξ) rtol = testtol

            x′, _ = retract(x, ξ, 0)
            Δ′ = @inferred transport(Δ₁, x, ξ, 0, x′)
            @test norm(first(x) - first(x′)) ≈ 0 atol = testtol
            @test inner(x′, Δ′, Δ′) ≈ inner(x, Δ₁, Δ₁) rtol = testtol
        end

        αs = range(1e-4, 1; length = 100)

        @testset "Isometric transport" begin
            for α in αs
                x′, _ = retract(x, ξ, α)
                @test inner(x, Δ₁, Δ₂) ≈ inner(x′, transport(Δ₁, x, ξ, α, x′), transport(Δ₂, x, ξ, α, x′)) rtol = testtol
            end
        end

        @testset "Finite differences" begin
            αs, fs, dfs1, dfs2 = @inferred OptimKit.optimtest(fg, x; alpha = αs, retract = retract, inner = inner)
            @test norm(dfs1 - dfs2, Inf) ≈ 0 atol = 1e-3
        end
    end
end


