

using Test
using LinearAlgebra
using StaticArrays
using CUDA
using SpatiotemporalGPs

println("CUDA Functional: $(CUDA.functional())")

function rand_posdef(N)
    A = randn(N, N)
    return Symmetric(A' * A + I)
end

function cuda_create_problem(kt_order, xs, ys, F)

    # create grid points
    pts = vec([(@SVector [x, y]) for x in xs, y in ys])

    # create temporal kernel
    σt = F(2.1)
    lt = F(3.1)
    kt = Matern(kt_order - 1 / 2, σt, lt)

    # create spatial kernel
    σs = F(1.1)
    ls = F(2.1)
    ks = Matern(3 / 2, σs, ls)

    # create the problem
    ΔT = F(0.1)
    prob = CudaSTGPKFProblem(pts, ks, kt, ΔT)

    return prob
end

@testset "CUDA STGPKF" for F in (Float32, Float64), kt_order in 1:3

    @testset "CUDA STGPKF - create"  begin
        xs = 0.0:1.0:5.0
        ys = 0.0:1.0:3.0
        prob = cuda_create_problem(kt_order, xs, ys, F)

        @test length(prob.pts) == length(xs) * length(ys)
        @test size(prob.ss_model.Φ) == (kt_order, kt_order)
        @test typeof(prob.sqrt_K_gg) <: Symmetric{F, M} where {M <: CuArray{F, 2}}
    end



    @testset "CUDA STGPKF - initialize" begin
        xs = 0.0:1.0:5.0
        ys = 0.0:1.0:3.0
        prob = cuda_create_problem(kt_order, xs, ys, F)

        # initialize
        state_0_0 = stgpkf_initialize(prob)

        @test length(state_0_0.μ) == length(prob.pts) * STGPKF.dims(prob.ss_model)
        @test isnothing(STGPKF.checkdims(prob, state_0_0))
        mu = state_0_0.μ
        Sigma = get_Σ(state_0_0)

        @test typeof(mu) <: CuArray{F, 1}
        @test typeof(Sigma) <: Cholesky{F, M} where {M <: CuArray{F, 2}}

    end

    @testset "STGPKF - predict correct"  begin
        xs = F.(0.0:1.0:5.0)
        ys = F.(0.0:1.0:3.0)
        prob = cuda_create_problem(kt_order, xs, ys, F)

        Ng = length(prob.pts)
        nk = STGPKF.dims(prob.ss_model)
        Nstate = Ng * nk

        # initialize
        state_0_0 = stgpkf_initialize(prob)

        # predict
        state_1_0 = stgpkf_predict(prob, state_0_0)
        @test length(state_1_0.μ) == Nstate

        # the very first prediction should NOT change the state or the covariance
        @test state_1_0.μ≈state_0_0.μ atol=1e-4
        @test Matrix(get_Σ(state_1_0))≈Matrix(get_Σ(state_0_0)) atol=1e-4

        # correct
        pt = SVector{2, F}(maximum(xs) * rand(), maximum(ys) * rand()) # random point
        y = randn() # random measurement
        σm = 0.1 # measurement noise
        state_1_1 = stgpkf_correct(prob, state_1_0, pt, y, σm)
        @test length(state_1_1.μ) == Nstate
        # check that the states are changed
        @test state_1_1.μ != state_1_0.μ
        @test state_1_1.U != state_1_0.U
        # @show typeof(state_1_1.μ)
        # @show typeof(state_1_1.U)


        # predict again
        state_2_1 = stgpkf_predict(prob, state_1_1)
        @test length(state_2_1.μ) == Nstate
        # check that the states are changed
        @test state_2_1.μ != state_1_1.μ
        @test state_2_1.U != state_1_1.U

        # correct again, but this time with multiple measurements
        N_measure = 10
        pts = [(@SVector [maximum(xs) * rand(), maximum(ys) * rand()]) for i in 1:N_measure] # random points
        ys = randn(N_measure) # random measurement
        Σm = rand_posdef(N_measure)

        state_2_2 = stgpkf_correct(prob, state_2_1, pts, ys, Σm)

        # check dimensions
        @test length(state_2_2.μ) == Nstate

        # check that the states are changed
        @test state_2_2.μ != state_2_1.μ
        @test state_2_2.U != state_2_1.U
    end

end