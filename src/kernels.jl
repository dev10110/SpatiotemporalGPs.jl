
"""
    SquaredExponential(σ, l)

creates a Squared Exponential kernel with variance `σ` and lengthscale `l`. Returns a `SqExp` kernel.
"""
function SquaredExponential(σ, l)
    return SqExp(σ^2, 1 / l)
end

"""
    Matern(order, σ, l)

creates a Matern kernel with order `order`, variance `σ`, and lengthscale `l`. 
Order must be (1/2, 3/2, or 5/2). Returns a `Matern12`, `Matern32`, or `Matern52` kernel.
"""
function Matern(order, σ, l)
    if (order == 1 / 2)
        return Matern12(σ^2, 1 / l)
    elseif (order == 3 / 2)
        return Matern32(σ^2, 1 / l)
    elseif (order == 5 / 2)
        return Matern52(σ^2, 1 / l)
    else
        throw(ArgumentError("Invalid Matern order. Must be 1/2, 3/2, or 5/2"))
    end
end

function dims(SS::ContinuousTimeStateSpaceModel)
    return size(SS.A, 1)
end
function dims(SS::DiscreteTimeStateSpaceModel)
    return size(SS.Φ, 1)
end

# define the kernel functions
function (k::SqExp)(x, y)
    d = norm(x - y)
    return k.σsq * exp(-0.5 * (k.λ * d)^2)
end

function (k::Matern12)(x, y)
    d = norm(x - y)
    return k.σsq * exp(-k.λ * d)
end

function (k::Matern32)(x, y)
    d = norm(x - y)
    return k.σsq * (1 + sqrt(3) * k.λ * d) * exp(-sqrt(3) * k.λ * d)
end

function (k::Matern52)(x, y)
    d = norm(x - y)
    return k.σsq * (1 + sqrt(5) * k.λ * d + 5 / 3 * (k.λ * d)^2) * exp(-sqrt(5) * k.λ * d)
end

function kernel_matrix(
        kernel::KK,
        X::VPX
) where {PX, VPX <: AbstractVector{PX}, KK <: AbstractKernel}
    NX = length(X)
    K = Array{Float64, 2}(undef, NX, NX)
    # only compute the upper triangle
    for i in 1:NX, j in i:NX
        K[i, j] = kernel(X[i], X[j])
    end
    return Symmetric(K)
end

"""
    kernel_matrix(kernel, X, Y)

Compute the kernel matrix between two sets of points `X` and `Y` using the kernel function `kernel`.
`X` must be a vector of points
`Y` must be a vector of points
"""
function kernel_matrix(
        kernel::KK,
        X::VPX,
        Y::VPY
) where {
        PX, PY, VPX <: AbstractVector{PX}, VPY <: AbstractVector{PY}, KK <: AbstractKernel}
    NX = length(X)
    NY = length(Y)
    K = Array{Float64, 2}(undef, NX, NY)
    for i in 1:NX, j in 1:NY
        K[i, j] = kernel(X[i], Y[j])
    end
    return K
end

## create the state-space representation
function state_space_model(kernel::Matern12{F}) where {F}
    λ = kernel.λ
    σ = sqrt(kernel.σsq)
    A = @SMatrix [-λ]
    B = @SMatrix [one(F)]
    C = @SMatrix [σ * sqrt(2λ)]

    SS = ContinuousTimeStateSpaceModel(A, B, C)

    return SS
end

function state_space_model(kernel::Matern32{F}) where {F}
    λ = kernel.λ
    σ = sqrt(kernel.σsq)
    A = @SMatrix [[zero(F);; one(F)]; [-3 * λ^2;; -2 * sqrt(3) * λ]]
    B = @SMatrix [[zero(F);;]; [one(F);;]]
    C = @SMatrix [σ * sqrt(12 * sqrt(3)) * λ^(3 / 2);; zero(F)]

    SS = ContinuousTimeStateSpaceModel(A, B, C)
    return SS
end

function state_space_model(kernel::Matern52{F}) where {F}
    λ = kernel.λ
    σ = sqrt(kernel.σsq)

    z = zero(F)
    o = one(F)

    A = @SMatrix [[z;; o;; z];
                  [z;; z;; o];
                  [-5 * sqrt(5) * λ^3;; -15 * λ^2;; -3 * sqrt(5) * λ]]

    B = @SMatrix [[z;;];
                  [z;;];
                  [o;;]]

    C = @SMatrix [[sqrt(400 * sqrt(5) / 3) * σ * λ^(5 / 2);; z;; z];]

    SS = ContinuousTimeStateSpaceModel(A, B, C)
    return SS
end

# get the discrete time state-space models
function state_space_model(kernel::Matern12, T)
    σ = sqrt(kernel.σsq)
    λ = kernel.λ

    Φ = @SMatrix [[exp(-T * λ);;];]
    W = @SMatrix [[-((-1 + exp(-2 * T * λ)) / (2 * λ));;];]
    C = @SMatrix [[σ * sqrt(2 * λ);;];]

    SS = DiscreteTimeStateSpaceModel(Φ, W, C, T)

    return SS
end

function state_space_model(kernel::Matern32, T)
    σ = sqrt(kernel.σsq)
    λ = kernel.λ

    s = sqrt(3) * λ * T
    ems = exp(-s)
    Φ = @SMatrix [[ems * (1 + s);; ems * T];
                  [-3 * ems * T * λ^2;; ems * (1 - s)]]

    w11 = (sqrt(3) - exp(-2 * s) * (sqrt(3) + 6 * T * λ * (1 + s))) / (36 * λ^3)
    w12 = 1 / 2 * exp(-2 * s) * T^2
    w21 = w12
    w22 = (exp(-2 * s) * (-1 + exp(2 * s) + 2 * T * λ * (sqrt(3) - 3 * T * λ))) /
          (4 * sqrt(3) * λ)

    W = @SMatrix [[w11;; w12]; [w21;; w22]]

    C = @SMatrix [[2 * 3^(3 / 4) * λ^(3 / 2) * σ;; 0];]

    SS = DiscreteTimeStateSpaceModel(Φ, W, C, T)

    return SS
end

function state_space_model(kernel::Matern52, T)
    σ = sqrt(kernel.σsq)
    λ = kernel.λ

    s = sqrt(5) * λ * T
    ems = exp(-s)

    Φ11 = ems * (1 + s + (5 * T^2 * λ^2) / 2)
    Φ12 = ems * (T + sqrt(5) * T^2 * λ)
    Φ13 = 1 / 2 * ems * T^2
    Φ21 = -(5 / 2) * sqrt(5) * ems * T^2 * λ^3
    Φ22 = ems * (1 + s - 5 * T^2 * λ^2)
    Φ23 = ems * (T - 1 / 2 * sqrt(5) * T^2 * λ)
    Φ31 = ems * (-5 * sqrt(5) * T * λ^3 + (25 * T^2 * λ^4) / 2)
    Φ32 = ems * (-15 * T * λ^2 + 5 * sqrt(5) * T^2 * λ^3)
    Φ33 = ems * (1 - 2 * s + (5 * T^2 * λ^2) / 2)

    Φ = @SMatrix [[Φ11;; Φ12;; Φ13];
                  [Φ21;; Φ22;; Φ23];
                  [Φ31;; Φ32;; Φ33]]

    W11 = (
        3 * sqrt(5) +
        exp(-2s) *
        (-3 * sqrt(5) - 10 * T * λ * (3 + T * λ * (3 * sqrt(5) + 5 * T * λ * (2 + s))))
    ) / (2000 * λ^5)

    W12 = 1 / 8 * exp(-2 * s) * T^4
    W13 = (
        exp(-2 * s) * (
        sqrt(5) - sqrt(5) * exp(2 * s) +
        10 * T * λ * (1 + T * λ * (sqrt(5) - 5 * T * λ * (-2 + s)))
    )
    ) / (400 * λ^3)
    W21 = W12
    W22 = (
        sqrt(5) +
        exp(-2 * s) *
        (-sqrt(5) - 10 * T * λ * (1 + T * λ * (sqrt(5) + 5 * T * λ * (-2 + s))))
    ) / (400 * λ^3)

    W23 = 1 / 8 * exp(-2 * s) * T^2 * (4 + T * λ * (-4 * sqrt(5) + 5 * T * λ))
    W31 = W13
    W32 = W23
    W33 = (
        3 * sqrt(5) +
        exp(-2 * s) *
        (-3 * sqrt(5) +
         10 * T * λ * (5 + T * λ * (-11 * sqrt(5) - 5 * T * λ * (-6 + s))))
    ) / (80 * λ)

    W = @SMatrix [[W11;; W12;; W13];
                  [W21;; W22;; W23];
                  [W31;; W32;; W33]]

    C = @SMatrix [[(20 * 5^(1 / 4) * λ^(5 / 2) * σ) / sqrt(3);; 0;; 0];]

    SS = DiscreteTimeStateSpaceModel(Φ, W, C, T)

    return SS
end

# get the initial covariance matrix
function initial_covariance(kernel::Matern12)
    λ = kernel.λ
    return @SMatrix [[1 / (2 * λ);;];]
end

function initial_covariance(kernel::Matern32)
    λ = kernel.λ
    return @SMatrix [[1 / (12 * sqrt(3) * λ^3);; 0];
                     [0;; 1 / (4 * sqrt(3) * λ)]]
end

function initial_covariance(kernel::Matern52)
    λ = kernel.λ
    Σ11 = 3 / (400 * sqrt(5) * λ^5)
    Σ12 = 0
    Σ13 = -(1 / (80 * sqrt(5) * λ^3))
    Σ21 = Σ12
    Σ22 = 1 / (80 * sqrt(5) * λ^3)
    Σ23 = 0
    Σ31 = Σ13
    Σ32 = Σ23
    Σ33 = 3 / (16 * sqrt(5) * λ)

    Σ = @SMatrix [[Σ11;; Σ12;; Σ13];
                  [Σ21;; Σ22;; Σ23];
                  [Σ31;; Σ32;; Σ33]]

    return Σ
end
