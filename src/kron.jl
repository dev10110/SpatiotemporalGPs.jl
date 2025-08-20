# some small fast utilities for the kronecker product

"""
    KroneckerIdentityProduct(B, N)

Represents the Kronecker product (I(N) ⊗ B) without forming the product.
"""
struct KroneckerIdentityProduct{T, TB} <: AbstractKroneckerProduct{T}
    B::TB
    N::Int
    function KroneckerIdentityProduct(B::TB, N::Int) where {TB <: AbstractMatrix}
        @assert N > 0
        return new{eltype(B), TB}(B, N)
    end
end

function Kronecker.getmatrices(K::KroneckerIdentityProduct)
    return (I(K.N), K.B)
end

function Kronecker.kronecker(A::Diagonal{Bool}, B::AbstractMatrix)
    # A is a diagonal matrix with boolean values
    # this is a special case where we can avoid forming the Kronecker product
    return KroneckerIdentityProduct(B, size(A, 1))
end

function Base.adjoint(K::KroneckerIdentityProduct)
    # adjoint of (I(N) ⊗ B) is (I(N) ⊗ B')
    # do a copy to force the adjoint to be evaluated
    return KroneckerIdentityProduct(copy(adjoint(K.B)), K.N)
end

function Base.transpose(K::KroneckerIdentityProduct)
    # transpose of (I(N) ⊗ B) is (I(N) ⊗ B')
    # do a copy to force the transpose to be evaluated
    return KroneckerIdentityProduct(copy(transpose(K.B)), K.N)
end

"""
    kron_I_B_mm!(Y, N, B, X)

In-place matrix version:
    Y := (I(N) ⊗ B) * X

- B :: m×q
- X :: (q*N)×K   (K right-hand sides)
- Y :: (m*N)×K
Works on Array and CuArray.
"""
function kron_I_B_mm!(Y::AbstractMatrix, N::Int, B::AbstractMatrix, X::AbstractMatrix)
    m, q = size(B)
    K = size(X,2)
    @assert N * q == size(X,1)
    @assert size(Y,1) == m*N && size(Y,2) == K

    # Fuse the N blocks: (qN × K) → (q × (N*K)), multiply, then reshape back
    Xr = reshape(X, q, N*K)
    Yr = reshape(Y, m, N*K)
    mul!(Yr, B, Xr)           
    return Y
end


"""
    kron_I_B_mv!(y, B, x)

In-place version: y := (I(N) ⊗ B) * x
y :: length m*N
"""
function kron_I_B_mv!(y::AbstractVector, N::Int, B::AbstractMatrix, x::AbstractVector)
    m, q = size(B)
    @assert N * q == length(x)
    @assert length(y) == m * N
    X = reshape(x, q, N)
    Y = reshape(y, m, N)
    mul!(Y, B, X)             
    return y
end


function LinearAlgebra.mul!(y::AbstractVector, K::KroneckerIdentityProduct, x::AbstractVector)
    kron_I_B_mv!(y, K.N, K.B, x)
    return y
end

function LinearAlgebra.mul!(Y::AbstractMatrix, K::KroneckerIdentityProduct, X::AbstractMatrix)
    kron_I_B_mm!(Y, K.N, K.B, X)
    return Y
end

function Base.:*(M::UpperTriangular{F, C}, K::KroneckerIdentityProduct) where {F, C <: AbstractMatrix{F}}
    # force it to become a normal matrix (sad - we loose the triangular nature of the output)
    return C(M) * K
end
function Base.:*(M::LowerTriangular{F, C}, K::KroneckerIdentityProduct) where {F, C <: AbstractMatrix{F}}
    # force it to become a normal matrix (sad - we loose the triangular nature of the output)
    return C(M) * K
end



function KF.chol_sqrt(K::KroneckerIdentityProduct)
    # chol_sqrt = cholesky(K).U
    # and cholesky(I ⊗ B).U =  I ⊗ (chol(B).U)
    # println("im here at chol_sqrt 147")
    return KroneckerIdentityProduct(cholesky(K.B).U, K.N)
end