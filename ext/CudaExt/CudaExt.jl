module CudaExt

using SpatiotemporalGPs

using Adapt, CUDA
using LinearAlgebra, StaticArrays, Kronecker
import SpecialFunctions
import Interpolations

STGPKF = SpatiotemporalGPs.STGPKF
KF = STGPKF.KF
AbstractKernel = STGPKF.AbstractKernel
KroneckerIdentityProduct = STGPKF.KroneckerIdentityProduct


include("mul_utils.jl")
include("cuda_kron.jl")
    

function SpatiotemporalGPs.CudaSTGPKFProblem(pts, ks::KS, kt::KT, ΔT::F) where {F, KS<: AbstractKernel{F}, KT <: AbstractKernel{F}}
    @assert length(pts)>0 "The grid points must be non-empty."

    # throw a warning if F != Float32
    F == Float32  || 
        @warn "Using CUDASTGPKFProblem with non-Float32 types may lead to performance issues. Consider using Float32."

    ss_model = cuda_state_space_model(kt, ΔT)

    # cost at problem creation
    K_gg = kernel_matrix(ks, pts) # is a Symmetric matrix
    sqrt_K_gg = Symmetric(sqrt(K_gg))
    inv_sqrt_K_gg = Symmetric(inv(sqrt_K_gg)) # might need to be smarter here about how to do inverse - maybe force chol first?

    # convert to CuArrays and Float32
    cu_sqrt_K_gg = adapt(CuArray, sqrt_K_gg)
    cu_inv_sqrt_K_gg = adapt(CuArray, inv_sqrt_K_gg)

    return CudaSTGPKFProblem(pts, ks, kt, ΔT, ss_model, cu_sqrt_K_gg, cu_inv_sqrt_K_gg)
end

#cuda version
function SpatiotemporalGPs.stgpkf_initialize(problem::CudaSTGPKFProblem{F}) where {F}

    grid_pts = problem.pts
    spatial_kernel = problem.ks
    temporal_kernel = problem.kt
    sampling_period = problem.ΔT

    # number of grid points
    Ng = length(grid_pts)

    # number of states in the state space model
    nk = STGPKF.ss_dims(temporal_kernel)

    # create the initial state
    x0 = zeros(F, nk * Ng) # everything starts at 0

    # create the covariance matrix
    P0 = STGPKF.initial_covariance(temporal_kernel)
    Σ0 = collect(I(Ng) ⊗ P0)

    # convert to CuArrays
    cu_x0 = adapt(CuArray, x0)
    cu_Σ0 = adapt(CuArray, Σ0)

    return KFState(; μ = cu_x0, Σ = cu_Σ0)
end


function cuda_state_space_model(kt::AbstractKernel{F}, ΔT::F) where {F}
    # create the state space model for the temporal kernel

    # construct on CPU
    dss= STGPKF.state_space_model(kt, ΔT)

    # now convert to CuArrays (maintains type)
    Φ = CuArray(dss.Φ)
    W = CuArray(dss.W)
    C = CuArray(dss.C)

    return STGPKF.DiscreteTimeStateSpaceModel(Φ, W, C, ΔT)
end

function SpatiotemporalGPs.stgpkf_correct(prob::CudaSTGPKFProblem{F},
        state::KFState,
        pts::VP,
        ys::VF2,
        Σm::MF2) where {
        P, F, F2, VP <: AbstractVector{P}, VF2 <: AbstractVector{F2}, MF2 <: AbstractMatrix{F2}}

    # get the number of grid points
    Ng = length(prob.pts)
    Nm = length(pts)

    # check the passed in prob and state are compatible dimensions
    STGPKF.checkdims(prob, state)

    # check that the measurements are of compatible dimensions  
    @assert length(ys)==Nm "The number of points and measurements must match."
    @assert size(Σm)==(Nm, Nm) "The measurement noise matrix must be of size (Nm, Nm)."
    # @assert isposdef(Σm) "Σm must be positive definite. remember to check `issymmetric(Σm)` is true."

    # construct the spatial kernel matrices # TODO: CUDA-ify
    K_mm = CuArray{F}(STGPKF.kernel_matrix(prob.ks, pts))
    K_mg = CuArray{F}(STGPKF.kernel_matrix(prob.ks, pts, prob.pts))

    # measurement matrix for a single grid point
    C = prob.ss_model.C

    # construct the measurement matrix for the full state
    L = K_mg * prob.inv_sqrt_K_gg
    H = L * (I(Ng) ⊗ C)

    # construct the noise matrix
    V = CuArray{F}(Σm) + K_mm - (L * L')


    # do the update
    new_state = KF.correct(state, CuVector{F}(ys), CuArray{F}(H), V)

    return new_state
end

function KF.qrr(
    A::Transpose{F, C},
    B::STGPKF.KroneckerIdentityProduct{F, UpperTriangular{F, C}}) where {F, C <: CuArray{F}}

    # println("im at KF.qrr on 134 with types $(typeof(A)) and $(typeof(B))")

    A_dense = copy(A) # force the transpose to be evaluated
    # println("A_dense: $(typeof(A_dense))")
    B_dense = make_dense(B)
    # println("B_dense: $(typeof(B_dense))")
    return KF.qrr(A_dense, B_dense)
end

function make_dense(K::KroneckerIdentityProduct{F, C}) where {F, C <: AbstractMatrix{F}}
    # println("converting K::KroneckerIdentityProduct to CuArray at 141")
    # materialize the KroneckerIdentityProduct as a CuArray
    n, m = size(K)
    K_dense = CuArray{F}(undef, (n, m))
    I_dense = CuArray{F}(I(m))
    mul!(K_dense, K, I_dense)  # materialize the KroneckerIdentityProduct
    # collect!(K_dense, K)  # materialize the KroneckerIdentityProduct
    return K_dense
end


end