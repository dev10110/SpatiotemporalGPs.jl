

abstract type AbstactKernel end
abstract type AbstractMaternKernel <: AbstactKernel end


# structs to hold hyperparameters
struct Matern12{F} <: AbstractMaternKernel
  σsq::F
  λ::F
end

struct Matern32{F} <: AbstractMaternKernel
  σsq::F
  λ::F
end

struct Matern52{F} <: AbstractMaternKernel
  σsq::F
  λ::F
end
"""
    Matern(order, σ, l)

creates a Matern kernel with order `order`, variance `σ`, and lengthscale `l`. 
Order must be (1/2, 3/2, or 5/2).
"""
function Matern(order, σ, l)
  if (order == 1/2) return Matern12(σ^2, 1/l)
  elseif (order == 3/2) return Matern32(σ^2, 1/l)
  elseif (order == 5/2) return Matern52(σ^2, 1/l)
  else
    throw(ArgumentError("Invalid Matern order. Must be 1/2, 3/2, or 5/2"))
  end
end

function (k::Matern12)(x, y)
  d = norm(x - y)
  return k.σsq * exp(- k.λ * d)
end

function (k::Matern32)(x, y)
  d = norm(x - y)
  return k.σsq * (1 + sqrt(3) * k.λ * d) * exp(- sqrt(3) * k.λ * d)
end

function (k::Matern52)(x, y)
  d = norm(x - y)
  return k.σsq * (1 + sqrt(5) * k.λ * d + 5/3 * (k.λ * d)^2) * exp(- sqrt(5) * k.λ * d)
end


"""
    kernel_matrix(kernel, X, Y)

Compute the kernel matrix between two sets of points `X` and `Y` using the kernel function `kernel`.
`X` must be a vector of points
`Y` must be a vector of points
"""
function kernel_matrix(kernel::KK, X::VP, Y::VP) where {P, VP<:AbstractVector{P}, KK <: AbstactKernel}
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
  A = @SMatrix [ -λ ]
  B = @SMatrix [ one(F) ]
  C = @SMatrix [ σ * sqrt(2λ)]

  return A, B, C

end

function state_space_model(kernel::Matern32{F}) where {F}

  λ = kernel.λ
  σ = sqrt(kernel.σsq)
  A = @SMatrix [ [zero(F);; one(F) ]; [-3 * λ^2 ;; -2*sqrt(3) * λ] ]
  B = @SMatrix [ [zero(F);; ]; [one(F);;]]
  C = @SMatrix [ σ * sqrt(12 * sqrt(3)) * λ^(3/2) ;; zero(F)]

  return A, B, C

end

function state_space_model(kernel::Matern52{F}) where  {F}

  λ = kernel.λ
  σ = sqrt(kernel.σsq)

  z = zero(F)
  o = one(F)

  A = @SMatrix [
    [ z ;; o ;; z];
    [ z ;; z ;; o];
    [ -5*sqrt(5) * λ^3 ;; -15*λ^2;; -3 * sqrt(5) * λ]
  ]

  B = @SMatrix [
    [ z ;;]; 
    [ z ;;]; 
    [ o ;;]; 
  ]

  C = @SMatrix [ [ sqrt(400 * sqrt(5) / 3) * σ * λ^(5/2) ;; z ;; z];]

  return A, B, C

end
  