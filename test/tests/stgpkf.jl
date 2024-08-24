
using SpatiotemporalGPs
using Test
using LinearAlgebra
using StaticArrays

function rand_posdef(N)
    A = randn(N, N)
    return Symmetric(A' * A + I)
end

function create_problem(kt_order, xs = 0.0:1.0:5.0, ys = 0.0:1.0:5.0)

    # create grid points
    pts = vec([(@SVector [x, y]) for x in xs, y in ys])

    # create temporal kernel
    σt = 2.1
    lt = 3.1
    kt = Matern(kt_order - 1 / 2, σt, lt)

    # create spatial kernel
    σs = 1.1
    ls = 2.1
    ks = Matern(3 / 2, σs, ls)

    # create the problem
    ΔT = 0.1
    prob = STGPKFProblem(pts, ks, kt, ΔT)

    return prob
end

@testset "STGPKF - create" for kt_order in 1:3
    xs = 0.0:1.0:5.0
    ys = 0.0:1.0:5.0
    prob = create_problem(kt_order, xs, ys)

    @test length(prob.pts) == length(xs) * length(ys)
    @test size(prob.ss_model.Φ) == (kt_order, kt_order)
end

@testset "STGPKF - initialize" for kt_order in 1:3
    prob = create_problem(kt_order)

    # initialize
    state_0_0 = stgpkf_initialize(prob)

    @test length(state_0_0.μ) == length(prob.pts) * STGPKF.dims(prob.ss_model)
    @test isnothing(STGPKF.checkdims(prob, state_0_0))
end

@testset "STGPKF - predict correct" for kt_order in 1:3
    xs = 0.0:1.0:5.0
    ys = 0.0:1.0:5.0
    prob = create_problem(kt_order, xs, ys)

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
    pt = @SVector [maximum(xs) * rand(), maximum(ys) * rand()] # random point
    y = randn() # random measurement
    σm = 0.1 # measurement noise
    state_1_1 = stgpkf_correct(prob, state_1_0, pt, y, σm)
    @test length(state_1_1.μ) == Nstate
    # check that the states are changed
    @test state_1_1.μ != state_1_0.μ
    @test state_1_1.U != state_1_0.U

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
