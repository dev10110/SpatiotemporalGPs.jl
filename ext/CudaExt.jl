module CudaExt

using SpatiotemporalGPs

using Adapt, CUDA
using LinearAlgebra, StaticArrays, Kronecker
import SpecialFunctions
import Interpolations

STGPKF = SpatiotemporalGPs.STGPKF
AbstractKernel = STGPKF.AbstractKernel
    
# provide a overload for A * B' that works with CUDA
function LinearAlgebra.mul!(C::CuArray{F, 2, M},
    A::UpperTriangular{F, CuArray{F, 2, M}},
    B::Adjoint{F, CuArray{F, 2, M}}) where {F, M}

    # force the copy when running with adjoint
    return mul!(C, A, copy(B) )
end


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

end