
using SpatiotemporalGPs
using Test
using LinearAlgebra
using StaticArrays

@testset "Kernels - create" begin

    σ = 5.0
    l = 3.0
    k12 = STGPKF.Matern(1/2, σ, l)
    k32 = STGPKF.Matern(3/2, σ, l)
    k52 = STGPKF.Matern(5/2, σ, l)

    @test k12.σsq ≈ σ^2
    @test k12.λ ≈ 1/l

    @test k32.σsq ≈ σ^2
    @test k32.λ ≈ 1/l

    @test k52.σsq ≈ σ^2
    @test k52.λ ≈ 1/l
end

@testset "Kernels - convert to state-space" begin

    for p=1:3

        σ = rand()
        l = rand()
        ν = p - 1/2 # order
        k = STGPKF.Matern(ν, σ, l)

        @test k.σsq ≈ σ^2
        @test k.λ ≈ 1/l

        A, B, C = STGPKF.state_space_model(k)

        @test size(A) == (p,p)
        @test size(B) == (p,1)
        @test size(C) == (1,p)
    end
end