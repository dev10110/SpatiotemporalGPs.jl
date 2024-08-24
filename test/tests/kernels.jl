
using SpatiotemporalGPs
using Test
using LinearAlgebra
using StaticArrays

@testset "Kernels - create" begin
    σ = 5.0
    l = 3.0
    k12 = STGPKF.Matern(1 / 2, σ, l)
    k32 = STGPKF.Matern(3 / 2, σ, l)
    k52 = STGPKF.Matern(5 / 2, σ, l)

    @test k12.σsq ≈ σ^2
    @test k12.λ ≈ 1 / l

    @test k32.σsq ≈ σ^2
    @test k32.λ ≈ 1 / l

    @test k52.σsq ≈ σ^2
    @test k52.λ ≈ 1 / l
end

@testset "Kernels - kernel_matrix" for order in 1:3
    σ = 5.0
    l = 3.0

    k = STGPKF.Matern(order - 1 / 2, σ, l)

    x = [@SVector randn(2) for i in 1:10]

    Kxx = kernel_matrix(k, x, x)

    @test size(Kxx) == (10, 10)
    @test Kxx ≈ Kxx'
    @test isposdef(Kxx)
end

@testset "Kernels - kernel_matrix 2" for order in 1:3
    σ = 5.0
    l = 3.0

    k = STGPKF.Matern(order - 1 / 2, σ, l)

    x = [@SVector randn(2) for i in 1:10]
    y = [@SVector randn(2) for i in 1:1]

    Kxy = kernel_matrix(k, x, y)

    @test size(Kxy) == (10, 1)
end

@testset "Kernels - convert to state-space" for p in 1:3
    σ = rand()
    l = rand()
    ν = p - 1 / 2 # order
    k = STGPKF.Matern(ν, σ, l)

    @test k.σsq ≈ σ^2
    @test k.λ ≈ 1 / l

    ss = STGPKF.state_space_model(k)

    @test size(ss.A) == (p, p)
    @test size(ss.B) == (p, 1)
    @test size(ss.C) == (1, p)
end

@testset "Kernels - analytic expressions of Matern12" begin
    λ = 3.2
    σ = 5.2
    T = 0.1

    k = STGPKF.Matern(1 / 2, σ, 1 / λ)

    # check continuous time
    ss = STGPKF.state_space_model(k)

    A_true = @SMatrix [[-3.2;;];]
    B_true = @SMatrix [[1.0;;];]
    C_true = @SMatrix [[13.1551;;];]

    @test ss.A≈A_true atol=1e-4
    @test ss.B≈B_true atol=1e-4
    @test ss.C≈C_true atol=1e-4

    # check discrete time
    dss = STGPKF.state_space_model(k, T)

    Φ_true = @SMatrix [[0.726149;;];]
    W_true = @SMatrix [[0.0738606;;];]
    H_true = @SMatrix [[13.1551;;];]

    @test dss.Φ≈Φ_true atol=1e-4
    @test dss.W≈W_true atol=1e-4
    @test dss.C≈H_true atol=1e-4
    @test dss.dt ≈ T
end

@testset "Kernels - analytic expressions of Matern32" begin
    λ = 6.3
    σ = 2.5
    T = 0.6

    k = STGPKF.Matern(3 / 2, σ, 1 / λ)

    # check continuous time
    ss = STGPKF.state_space_model(k)

    A_true = @SMatrix [[0;; 1];
                       [-119.07;; -21.8238]]
    B_true = @SMatrix [[0;;]; [1.0;;]]
    C_true = @SMatrix [[180.22779508447948;; 0];]

    @test ss.A≈A_true atol=1e-4
    @test ss.B≈B_true atol=1e-4
    @test ss.C≈C_true atol=1e-4

    # check discrete time
    dss = STGPKF.state_space_model(k, T)

    Φ_true = @SMatrix [[0.0108241;; 0.000860517];
                       [-0.102462;; -0.00795569]]
    W_true = @SMatrix [[0.000192374;; 3.70244 * 10^(-7)];
                       [3.70244 * 10^(-7);; 0.0229073]]
    H_true = @SMatrix [[180.22779508447948;; 0];]

    @test dss.Φ≈Φ_true atol=1e-4
    @test dss.W≈W_true atol=1e-4
    @test dss.C≈H_true atol=1e-4
    @test dss.dt ≈ T
end

@testset "Kernels - analytic expressions of Matern52" begin
    λ = 1.39
    σ = 1.84
    T = 1.6

    k = STGPKF.Matern(5 / 2, σ, 1 / λ)

    # check continuous time
    ss = STGPKF.state_space_model(k)

    A_true = @SMatrix [[0;; 1;; 0];
                       [0;; 0;; 1];
                       [-30.0261;; -28.9815;; -9.3244]]
    B_true = @SMatrix [[0;;]; [0;;]; [1.0;;]]
    C_true = @SMatrix [[72.37135318168716;; 0;; 0];]

    @test ss.A≈A_true atol=1e-4
    @test ss.B≈B_true atol=1e-4
    @test ss.C≈C_true atol=1e-4

    # check discrete time
    dss = STGPKF.state_space_model(k, T)

    Φ_true = @SMatrix [[0.126943;; 0.0661547;; 0.00886047];
                       [-0.266046;; -0.129847;; -0.016464];
                       [0.494349;; 0.211104;; 0.02367]]
    W_true = @SMatrix [[0.000626822;; 0.000039254;; -0.00214843];
                       [0.000039254;; 0.00200256;; 0.000135531];
                       [-0.00214843;; 0.000135531;; 0.0600898]]
    H_true = @SMatrix [[72.37135318168716;; 0;; 0];]

    @test dss.Φ≈Φ_true atol=1e-4
    @test dss.W≈W_true atol=1e-4
    @test dss.C≈H_true atol=1e-4
    @test dss.dt ≈ T
end
