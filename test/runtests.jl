using SpatiotemporalGPs
using Test
using LinearAlgebra

@testset "SpatiotemporalGPs.jl" begin
    
end

@testset "KalmanFilter - create (diagonal)" begin

    # try the diagonal case 
    x = [1.0, 2.0]
    P = diagm([1.0, 5.0]) 
    s = KFState(μ=x, Σ=P)

    @test μ(s) ≈ x
    @test Matrix(Σ(s)) ≈ P
    @test σ(s) ≈ sqrt.(diag(P))
end
