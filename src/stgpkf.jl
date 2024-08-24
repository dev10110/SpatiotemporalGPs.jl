module STGPKF
using LinearAlgebra
using StaticArrays
using Kronecker
import SpecialFunctions
import Interpolations
# this module creates a spatiotemporal GP Kalman Filter

export Matern, SquaredExponential
export kernel_matrix, state_space_model
export STGPKFProblem
export stgpkf_initialize, stgpkf_predict, stgpkf_correct
export generate_spatiotemporal_process

import ..KalmanFilter
KF = KalmanFilter
KFState = KF.KFState

include("types.jl")
include("kernels.jl")
include("plotting.jl")
include("synthetic.jl")

"""
    STGPKFProblem(pts, ks, kt, ΔT)

Defines a spatiotemporal Gaussian Process Kalman Filter problem. Parameters are:
  - `pts`: grid points, a vector of all points. Ideally, `eltype(pts)` should be `StaticVector` for efficiency 
  - `ks`: spatial kernel, must be of type `AbstractKernel`
  - `kt`: temporal kernel, must be of type `AbstractKernel` (but only AbstractMaternKernel is implemented)
  - `ΔT`: sampling period
"""
function STGPKFProblem(pts, ks, kt, ΔT)
    @assert length(pts)>0 "The grid points must be non-empty."

    ss_model = state_space_model(kt, ΔT)

    # cost at problem creation
    K_gg = kernel_matrix(ks, pts) # is a Symmetric matrix
    sqrt_K_gg = Symmetric(sqrt(K_gg))
    inv_sqrt_K_gg = Symmetric(inv(sqrt_K_gg))
    return STGPKFProblem(pts, ks, kt, ΔT, ss_model, sqrt_K_gg, inv_sqrt_K_gg)
end

"""
    checkdims(prob, state)
checks that the dimensions of the state and the problem match
"""
function checkdims(prob::STGPKFProblem, state::KFState)
    Ng = length(prob.pts)
    nk = dims(prob.ss_model)
    nx = length(state)
    @assert Ng * nk==nx "Dimensions of state and problem do not match"
end

""" 
    get_states(problem, state)
  
returns the states of the Kalman Filter for all grid points, in a Vector{SVector{F}} format. The outer vector has same length as `problem.pts`.
"""
function get_states(problem::STGPKFProblem, state::KFState)
    nk = dims(problem.ss_model)
    Ng = length(problem.pts)
    μ = KF.get_μ(state)
    x = [SVector{nk}(μ[((i - 1) * nk + 1):(i * nk)]) for i in 1:Ng]
    return x
end

"""
    get_marginal_states(problem, state)

returns the marginal states of the Kalman Filter for all grid points, in a Vector{KFState} format. The outer vector has same length as `problem.pts`.
"""
function get_marginal_states(problem::STGPKFProblem, state::KFState)
    nk = dims(problem.ss_model)
    Ng = length(problem.pts)
    μ = KF.get_μ(state)
    Σ = Matrix(KF.get_Σ(state))
    xs = [SVector{nk}(μ[((i - 1) * nk + 1):(i * nk)]) for i in 1:Ng]
    Σs = [SMatrix{nk, nk}(Σ[((i - 1) * nk + 1):(i * nk), ((i - 1) * nk + 1):(i * nk)])
          for i in 1:Ng]

    s = [KFState(; μ = xs[i], Σ = Σs[i]) for i in 1:Ng]
    return s
end

"""
    get_estimate(problem, state)

returns the estimate of the Kalman Filter for all grid points, in a Vector{F} format. The outer vector has same length as `problem.pts`.
"""
function get_estimate(problem::STGPKFProblem, state::KFState)
    Ng = length(problem.pts)
    C = problem.ss_model.C
    # spatially uncorrelated components
    z = (I(Ng) ⊗ C) * KF.get_μ(state)
    # correlate spatially using the K_gg matrix
    x = problem.sqrt_K_gg * z

    return x
end

"""
    get_estimate_covariance(problem, state)

returns the kalman filter's covariance of the estimated spatiotemporal field at all grid points, in a Vector{F} format.  The vector has same length as `problem.pts`.
"""
function get_estimate_covariance(problem::STGPKFProblem, state::KFState)
    Ng = length(problem.pts)
    C = problem.ss_model.C
    # L = problem.sqrt_K_gg * (I(Ng) ⊗ C)

    # much faster than 
    # Σ = L * Matrix(get_Σ(state)) * L'
    # when L is fat
    # z = state.U * L'
    # Σ = Symmetric(z' * z)

    zT = problem.sqrt_K_gg * ((I(Ng) ⊗ C) * state.U')
    Σ = Symmetric(zT * zT')
    return Σ
end

"""
    get_estimate_std(problem, state)

returns the standard deviation of the estimated spatiotemporal field at all grid points, in a Vector{F} format. The vector has same length as `problem.pts`.
"""
function get_estimate_std(problem::STGPKFProblem, state::KFState)
    Σ = get_estimate_covariance(problem, state)
    σ = sqrt.(diag(Σ))
    return σ
end

"""
    get_estimate_clarities(problem, state)

returns the clarity of the estimated spatiotemporal field at all grid points, in a Vector{F} format. The vector has same length as `problem.pts`.
"""
function get_estimate_clarity(problem::STGPKFProblem, state::KFState)
    Σ = get_estimate_covariance(problem, state)
    q = [1 / (1 + Σii) for Σii in diag(Σ)]
    return q
end

"""
    get_estimate_percentile(problem, state, percentile)

returns the percentile-% quantile of the estimated spatiotemporal field at all grid points, in a Vector{F} format. The vector has same length as `problem.pts`.
"""
function get_estimate_percentile(problem::STGPKFProblem, state::KFState, percentile)
    est = get_estimate(problem, state)
    σs = get_estimate_std(problem, state)
    return [quantile(μ, σ, percentile) for (μ, σ) in zip(est, σs)]
end

function spatial_interpolate(
        problem::STGPKFProblem, state::KFState, pts::VP) where {P, VP <: AbstractVector{P}}
    K_mg = kernel_matrix(problem.ks, pts, problem.pts)

    C = problem.ss_model.C
    L = K_mg * problem.inv_sqrt_K_gg
    H = L * (I(length(problem.pts)) ⊗ C)

    return H * KF.get_μ(state)
end

"""
    stgpkf_initialize(problem)
returns a  `KFState` that represents the initial state of the Kalman Filter for all grid points
"""
function stgpkf_initialize(problem::STGPKFProblem)
    grid_pts = problem.pts
    spatial_kernel = problem.ks
    temporal_kernel = problem.kt
    sampling_period = problem.ΔT

    # create the state-space model
    SS = state_space_model(temporal_kernel, sampling_period)

    # number of grid points
    Ng = length(grid_pts)
    # number of states in the state space model
    nk = dims(SS)

    # create the initial state
    x0 = zeros(nk * Ng) # everything starts at 0

    # create the covariance matrix
    P0 = initial_covariance(temporal_kernel)
    Σ0 = I(Ng) ⊗ P0
    kfState = KFState(; μ = x0, Σ = Σ0)
    return kfState
end

# construct the next state estimate by propagating the state space model by one timestep
"""
    stgpkf_predict(prob, state)
  
predicts the next state of the Kalman Filter for all grid points
"""
function stgpkf_predict(prob::STGPKFProblem, state::KFState)
    checkdims(prob, state)

    Ng = length(prob.pts)
    # A = I(Ng) ⊗ prob.ss_model.Φ
    # W = I(Ng) ⊗ prob.ss_model.W
    A = kron(I(Ng), prob.ss_model.Φ)
    W = kron(I(Ng), prob.ss_model.W)

    new_state = KF.predict(state, A, W)

    return new_state
end

"""
    stgpkf_correct(prob, state, pt, y, σ_m)

corrects the state of the Kalman Filter given a single point measurement at ``pt`` with value ``y`` and measurement noise standard deviation ``σ_m``.
"""
function stgpkf_correct(
        prob::STGPKFProblem{P, F}, state::KFState, pt::P, y::F, σ_m::F) where {P, F}
    vec_pts = [pt]
    vec_ys = @SVector [y]
    mat_Σm = @SMatrix [[σ_m^2;;];]
    return stgpkf_correct(prob, state, vec_pts, vec_ys, mat_Σm)
    return new_state
end

"""
    stgpkf_correct(prob, state, pts, ys, Σm)

corrects the state of the Kalman Filter given multiple point measurements at ``pts`` with values ``ys`` and measurement noise covariance matrix ``Σm``.
"""
function stgpkf_correct(prob::STGPKFProblem{P, F},
        state::KFState,
        pts::VP,
        ys::VF,
        Σm::MF) where {
        P, F, VP <: AbstractVector{P}, VF <: AbstractVector{F}, MF <: AbstractMatrix{F}}

    # get the number of grid points
    Ng = length(prob.pts)
    Nm = length(pts)

    # check the passed in prob and state are compatible dimensions
    checkdims(prob, state)

    # check that the measurements are of compatible dimensions  
    @assert length(ys)==Nm "The number of points and measurements must match."
    @assert size(Σm)==(Nm, Nm) "The measurement noise matrix must be of size (Nm, Nm)."
    @assert isposdef(Σm) "Σm must be positive definite. remember to check `issymmetric(Σm)` is true."

    # construct the spatial kernel matrices
    K_mm = kernel_matrix(prob.ks, pts)
    K_mg = kernel_matrix(prob.ks, pts, prob.pts)

    # measurement matrix for a single grid point
    C = prob.ss_model.C

    # construct the measurement matrix for the full state
    L = K_mg * prob.inv_sqrt_K_gg
    H = L * (I(Ng) ⊗ C)

    # construct the noise matrix 
    V = Symmetric(Σm) + Symmetric(K_mm) - Symmetric(L * L')

    # do the update
    new_state = KF.correct(state, ys, H, V)

    return new_state
end

end
