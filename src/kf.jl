### Kalman Filter Module
### Devansh Agrawal Mar 2024

# implementation of Kevin Tracy A SQUARE-ROOT KALMAN FILTER USING ONLY QR DECOMPOSITIONS
# https://arxiv.org/pdf/2208.06452.pdf
# also should look at https://ntrs.nasa.gov/api/citations/19770005172/downloads/19770005172.pdf


module KalmanFilter


export KFState, μ, Σ, σ, kalman_filter, predict, correct

using LinearAlgebra


"""
    KFState{V, M}

A type for the Kalman Filter State, which is parameterized by the types of the mean estimate and the upper triangular cholesky component of the covariance matrix.
"""
struct KFState{T,V<:AbstractVector{T},M,U<:UpperTriangular{T,M}}
    μ::V # mean estimate of the kalman filter
    F::U # upper triangular cholesky component of the Kalman State
end

#######################
## Getters and Setters
#######################

"""
    KFState(; μ, Σ, make_symmetric=true)

A constructor for the Kalman Filter State, which is parameterized by the mean estimate and the covariance matrix.
If `make_symmetric` is true, the covariance matrix is made symmetric internally. This is useful for numerical stability.
"""
function KFState(; μ, Σ, make_symmetric = true)
    Γ = make_symmetric ? chol_sqrt(Symmetric(Σ)) : chol_sqrt(Σ)
    return KFState(μ, Γ)
end

"""
    μ(s::S) where {S <: KFState}
Get the mean estimate of the Kalman Filter State.
"""
function μ(s::S) where {S<:KFState}
    return s.μ
end

"""
    Σ(s::S) where {S <: KFState}
Get the covariance matrix of the Kalman Filter State.
"""
function Σ(s::S) where {S<:KFState}
    return Cholesky(s.F) # avoids explicitly computing the full matrix, which could save computation
end

"""
    σ(s::S) where {S <: KFState}
Get a vector of the standard deviation of the Kalman Filter State
"""
function σ(s::S) where {S<:KFState}
    return sqrt.(diag(Σ(s)))
end

##################
# main methods
##################

"""
    s_{k+1} = kalman_filter(s_k, y_{k+1}, u_k, A, B, C, V, W)

Runs both the prediction and the correction steps. Assumes a system model
```math
  \\begin{align}
  x_{k+1} &= A x_k + B u_k + w, \\\\
  y_k &= C x_k + v
  \\end{align}
```
where ``w ∼ \\mathcal{N}(0, W)``, ``v ∼ \\mathcal{N}(0, V)``.
"""
function kalman_filter(s_k::S, y_kp1, u_k, A, B, C, V, W) where {S<:KFState}

    s_pred = predict(s_k, A, B, u_k, W)
    s_corr = correct(s_pred, y_kp1, C, V)
    return s_corr

end

"""
    s_{k+1|k} = predict(s_{k|k}, A, W)

Uses the system model
```math
  x_{k+1} = A x_k + w
```
where ``w ∼ N(0, W)`` to predict the next state.
"""
function predict(s::S, A, W) where {S<:KFState}

    N = length(s.μ)
    Γw = chol_sqrt(0 * I(N) + W)

    μ_new = A * s.μ
    F_new = qrr(s.F * A', Γw)

    return KFState(μ_new, F_new)

end

"""
      s_{k+1} = predict(s_k, A, B, u_k, W)

Uses the system model
```math
  x_{k+1} = A x_k + B u_k  + w
```
where ``w ∼ \\mathcal{N}(0, W)`` to predict the next state.
"""
function predict(s::S, A, B, u, W) where {S<:KFState}

    N = length(s.μ)
    Γw = chol_sqrt(0 * I(N) + W)

    μ_new = A * S.μ + B * u
    F_new = qrr(s.F * A', Γw)

    return KFState(μ_new, F_new)

end


"""
    s_{k+1|k+1} = correct(s_{k+1|k}, y_{k+1}, C, V)

Uses the system model
```math
y_{k+1} = C x_{k+1} + v
```
where ``v \\sim \\mathcal{N}(0, V)`` to correct the predicted state.
"""
function correct(s::S, y, C, V) where {S<:KFState}

    M = length(y)
    Γv = chol_sqrt(0 * I(M) + V)

    # innovation
    z = y - C * s.μ

    # kalman gain
    L = kalman_gain(s, C, Γv)

    # update
    μ_new = s.μ + L * z

    # @time sqrtA_ = s.F * (I - L * C)'
    sqrtA_ = s.F + (s.F * C') * (-L')
    sqrtB_ = Γv * L'

    F_new = qrr(sqrtA_, sqrtB_)

    return KFState(μ_new, F_new)

end


##############
# UTILITIES
##############

"""
    U = chol_sqrt(A)

returns an upper-triangular matrix ``U`` such that ``A = U^T U``.
"""
function chol_sqrt(A)
    return cholesky(A).U
end

"""
    R = qrr(A, B)

returns 
```math
R = \\sqrt{A^TA + B^TB}
```

The result is an `UpperTriangular` matrix.
"""
function qrr(sqrtA, sqrtB)

    M = [sqrtA; sqrtB]

    return qrr!(M)

end


function qrr!(A)
    N = minimum(size(A))
    LinearAlgebra.LAPACK.geqrf!(A) # TODO(dev): do this but use generic Julia rather than LAPACK
    return UpperTriangular(A[1:N, 1:N])
end

function kalman_gain(s::S, C, Γv) where {S<:KFState}

    G = qrr(s.F * C', Γv)
    L = ((s.F' * s.F * C') / G) / (G')

    return L

end



"""
    diag(M::Cholesky)

is a fast method for getting the diagonal of a cholesky matrix.

This will eventually be included into the Julia standard library. 
https://github.com/JuliaLang/julia/pull/53767
"""
function LinearAlgebra.diag(M::Cholesky{T}) where {T}
    N = size(M, 1)
    z = zeros(T, N)
    U = M.U
    for i = 1:N
        for j = 1:i
            z[i] += U[j, i]^2
        end
    end
    return z
end



end
