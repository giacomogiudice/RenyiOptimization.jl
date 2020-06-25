@testset "Testset for type $(T)" for T in (Float32, Float64, ComplexF32, ComplexF64)
    tol = 10*eps(real(T))

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

            @test ζ ≈ 1 atol = tol
            @test ρL ≈ one(ρL) atol = tol
            @test ρL ⊙ ρR ≈ 1 atol = tol
            @test righttransfer(ρR, AL) ≈ ρR atol = tol
            @test (∇ζ ⋅ AL)/2 ≈ 1 atol = tol
        end

        @testset "Double fixed points" begin
            ΣL, ΣR, η, ∇η = @inferred doublefixedpoints(AL; tol = tol)

            @test ΣL ⊙ ΣR ≈ 1 atol = tol
            @test norm(leftdoubletransfer(ΣL, AL) - η*ΣL, Inf) ≈ 0 atol = tol
            @test norm(rightdoubletransfer(ΣR, AL) - η*ΣR, Inf) ≈ 0 atol = tol
            @test (∇η ⋅ AL)/2 ≈ η atol = tol
        end

        @testset "Energy fixed points" begin
            ρL, ρR, _, _ = singlefixedpoints(AL; tol = tol)
            HL, HR, ε, ∇ε = @inferred energyfixedpoints(AL, H, ρL, ρR; tol = tol)

            @test HL ⊙ ρR ≈ 0 atol = tol
            @test ρL ⊙ HR ≈ 0 atol = tol
            @test ε ≈ expectationvalue(O, AL) atol = tol
            @test (∇ε ⋅ AL)/2 ≈ ε atol = tol
        end
    end 
end
