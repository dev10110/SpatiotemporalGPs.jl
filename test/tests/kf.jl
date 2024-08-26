
using SpatiotemporalGPs
using Test
using LinearAlgebra
using StaticArrays

function rand_posdef(N)
    A = randn(N, N)
    return Symmetric(A * A' + I)
end

@testset "KalmanFilter - create (diagonal)" begin

    # try the diagonal case 
    x = [1.0, 2.0]
    P = diagm([1.0, 5.0])
    s = KFState(μ = x, Σ = P)

    @test get_μ(s) ≈ x
    @test Matrix(get_Σ(s)) ≈ P
    @test get_σ(s) ≈ sqrt.(diag(P))
end

@testset "KalmanFilter - create Vectors" begin
    for N in 1:15
        x = randn(N)
        sqrtP = randn(N, N)
        P = sqrtP' * sqrtP + I

        s = KFState(μ = x, Σ = P)

        @test get_μ(s) ≈ x
        @test Matrix(get_Σ(s)) ≈ P
        @test get_σ(s) ≈ sqrt.(diag(P))
    end
end

@testset "KalmanFilter - throws PosDef" begin
    N = 5
    x = randn(N)
    sqrtP = randn(N, N)
    P1 = -(sqrtP' * sqrtP + I) # definetely a negative definite matrix

    @test_throws PosDefException KFState(μ = x, Σ = P1)

    P2 = diagm([1, 1, 1, 1, 0.0]) # not positive definite
    @test_throws PosDefException KFState(μ = x, Σ = P2)

    P3 = diagm([1, 1, 1, 1, -1.0]) # not positive definite
    @test_throws PosDefException KFState(μ = x, Σ = P3)
end

@testset "KalmanFilter - create SVectors" begin
    for N in 1:5
        x = @SVector randn(N)
        sqrtP = @SMatrix randn(N, N)
        P = sqrtP' * sqrtP + I

        s = KFState(μ = x, Σ = P)

        @test get_μ(s) ≈ x
        @test Matrix(get_Σ(s)) ≈ P
        @test get_σ(s) ≈ sqrt.(diag(P))
    end
end

@testset "KalmanFilter - predict" begin
    N = 4

    x = randn(N)
    sqrtP = randn(N, N)
    P = sqrtP * sqrtP' + I
    s = KFState(μ = x, Σ = P)

    # dynamics (integrator)
    A = [[zeros(N - 1);; I(N - 1)]; zeros(N)']
    dt = 0.1
    Ad = exp(A * dt) # convert to discrete time

    # process noise
    sqrtW = randn(N, N)
    W = sqrtW * sqrtW' + I

    # run the prediction
    s_new = predict(s, Ad, W)

    # test
    P_new = Ad * P * Ad' + W
    @test get_μ(s_new) ≈ Ad * x
    @test Matrix(get_Σ(s_new)) ≈ P_new
    @test get_σ(s_new) ≈ sqrt.(diag(P_new))
end

@testset "KalmanFilter - correct" begin
    N = 4 # number of states
    M = 2 # number of measurements

    x = randn(N)
    sqrtP = randn(N, N)
    P = sqrtP * sqrtP' + I
    s = KFState(μ = x, Σ = P)

    # measurement matrix
    C = randn(M, N)

    # measurement noise
    sqrtV = randn(M, M)
    V = sqrtV * sqrtV' + I

    # create a measurement
    y = C * x + randn(M)

    # run the correction
    s_new = correct(s, y, C, V)

    # test
    K = P * C' * inv(C * P * C' + V)
    P_new = (I - K * C) * P
    @test get_μ(s_new) ≈ x + K * (y - C * x)
    @test Matrix(get_Σ(s_new)) ≈ P_new
    @test get_σ(s_new) ≈ sqrt.(diag(P_new))
end

@testset "KalmanFilter - qrr" begin
    N = 20
    A = rand_posdef(N)
    B = rand_posdef(N)

    sqrtA = KalmanFilter.chol_sqrt(A) # upper triangular
    sqrtB = KalmanFilter.chol_sqrt(B) # upper triangular

    @test sqrtA' * sqrtA ≈ A
    @test sqrtB' * sqrtB ≈ B

    sqrt_R_true = KalmanFilter.chol_sqrt(A + B)

    sqrt_R = KalmanFilter.qrr(sqrtA, sqrtB)

    @test sqrt_R' * sqrt_R ≈ sqrt_R_true' * sqrt_R_true
end
