
using SpatiotemporalGPs
using Test
using LinearAlgebra
using StaticArrays

@testset "KalmanFilter - create (diagonal)" begin

    # try the diagonal case 
    x = [1.0, 2.0]
    P = diagm([1.0, 5.0]) 
    s = KFState(μ=x, Σ=P)

    @test μ(s) ≈ x
    @test Matrix(Σ(s)) ≈ P
    @test σ(s) ≈ sqrt.(diag(P))
end

@testset "KalmanFilter - create SVectors" begin

    for N=1:5
        x = @SVector randn(N)
        sqrtP = @SMatrix randn(N, N)
        P = sqrtP * sqrtP' + I
        s = KFState(μ=x, Σ=P)

        @test μ(s) ≈ x
        @test Matrix(Σ(s)) ≈ P
        @test σ(s) ≈ sqrt.(diag(P))
    end
end



@testset "KalmanFilter - predict" begin

    N = 4 # number of states
    
    x = randn(N)
    sqrtP =  randn(N, N)
    P = sqrtP * sqrtP' + I
    s = KFState(μ=x, Σ=P)

    # dynamics (integrator)
    A =  [ [zeros(N-1) ;; I(N-1)]; zeros(N)' ]
    dt = 0.1
    Ad = exp(A * dt) # convert to discrete time

    # process noise
    sqrtW = randn(N, N)
    W = sqrtW * sqrtW' + I

    # run the prediction
    s_new = predict(s, Ad, W)

    # test
    P_new = Ad * P * Ad' + W
    @test μ(s_new) ≈ Ad*x
    @test Matrix(Σ(s_new)) ≈ P_new
    @test σ(s_new) ≈ sqrt.(diag(P_new))
end

@testset "KalmanFilter - correct" begin

    N = 4 # number of states
    M = 2 # number of measurements
    
    x = randn(N)
    sqrtP =  randn(N, N)
    P = sqrtP * sqrtP' + I
    s = KFState(μ=x, Σ=P)

    # measurement matrix
    C = randn(M, N)

    # measurement noise
    sqrtV = randn(M, M)
    V = sqrtV * sqrtV' + I

    # create a measurement
    y = C*x + randn(M)

    # run the correction
    s_new = correct(s, y, C, V)

    # test
    K = P*C' * inv(C*P*C' + V)
    P_new = (I - K*C)*P
    @test μ(s_new) ≈ x + K*(y - C*x)
    @test Matrix(Σ(s_new)) ≈ P_new
    @test σ(s_new) ≈ sqrt.(diag(P_new))
end