
abstract type AbstractKernel{F} end
abstract type AbstractMaternKernel{F} <: AbstractKernel{F} end

struct SqExp{F} <: AbstractKernel{F}
    σsq::F
    λ::F
end

# structs to hold hyperparameters
struct Matern12{F} <: AbstractMaternKernel{F}
    σsq::F
    λ::F
end

struct Matern32{F} <: AbstractMaternKernel{F}
    σsq::F
    λ::F
end

struct Matern52{F} <: AbstractMaternKernel{F}
    σsq::F
    λ::F
end

abstract type AbstractStateSpaceModel end

struct ContinuousTimeStateSpaceModel{MA, MB, MC} <: AbstractStateSpaceModel
    A::MA
    B::MB
    C::MC
end

struct DiscreteTimeStateSpaceModel{MΦ, MW, MC, F} <: AbstractStateSpaceModel
    Φ::MΦ
    W::MW
    C::MC
    dt::F
end

abstract type AbstractSTGPKFProblem{F, P} end


struct STGPKFProblem{
    F,
    P,
    VP <: AbstractVector{P},
    KS <: AbstractKernel{F},
    KT <: AbstractKernel{F},
    DTSS <: DiscreteTimeStateSpaceModel,
    M1 <: AbstractMatrix{F},
    M2 <: AbstractMatrix{F}
} <: AbstractSTGPKFProblem{F, P}
    pts::VP # grid points
    ks::KS  # spatial kernel
    kt::KT # temporal kernel
    ΔT::F # sampling period
    ss_model::DTSS # state space model (discrete time) (for the temporal kernel)
    sqrt_K_gg::M1
    inv_sqrt_K_gg::M2 # inverse of the square root of the spatial kernel matrix
end

# create a similar struct for the Cuda version
struct CudaSTGPKFProblem{
    F,
    P,
    VP <: AbstractVector{P},
    KS <: AbstractKernel{F},
    KT <: AbstractKernel{F},
    DTSS <: DiscreteTimeStateSpaceModel,
    M1 <: AbstractMatrix{F},
    M2 <: AbstractMatrix{F}
} <: AbstractSTGPKFProblem{F, P}
    pts::VP # grid points
    ks::KS  # spatial kernel
    kt::KT # temporal kernel
    ΔT::F # sampling period
    ss_model::DTSS # state space model (discrete time) (for the temporal kernel)
    sqrt_K_gg::M1
    inv_sqrt_K_gg::M2 # inverse of the square root of the spatial kernel matrix
end

abstract type AbstractSpatiotemporalData end

struct SpatiotemporalData{P, F, VF <: AbstractVector{F}, AF <: AbstractArray{F}, TI} <:
       AbstractSpatiotemporalData
    grid_pts::Vector{P}
    ts::VF
    data::AF
    itp::TI
end

struct SpatiotemporalData2D{F, VF <: AbstractVector{F}, AF <: AbstractArray{F}, TI} <:
       AbstractSpatiotemporalData
    xs::VF
    ys::VF
    ts::VF
    data::AF
    itp::TI
end
