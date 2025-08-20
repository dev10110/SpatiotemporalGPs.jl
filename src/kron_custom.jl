# some small fast utilities for the kronecker product
"""
    KroneckerIdentityProduct(B, N)

Represents the Kronecker product (I(N) ⊗ B) without forming the product.
"""
struct KroneckerIdentityProduct{T, TB}  <: AbstractMatrix{T}
    B::TB
    N::Int
    function KroneckerIdentityProduct(B::TB, N::Int) where {TB <: AbstractMatrix}
        @assert N > 0
        return new{eltype(B), TB}(B, N)
    end
end


function ⊗(A::Diagonal{Bool}, B::AbstractMatrix)
    # A is a diagonal matrix with boolean values
    # this is a special case where we can avoid forming the Kronecker product
    return KroneckerIdentityProduct(B, size(A, 1))
end


function adjoint(K::KroneckerIdentityProduct)
    # adjoint of (I(N) ⊗ B) is (I(N) ⊗ B')
    return KroneckerIdentityProduct(adjoint(K.B), K.N)
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
    mul!(Yr, B, Xr)           # BLAS/cuBLAS GEMM
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
    mul!(Y, B, X)             # BLAS/cuBLAS GEMM
    return y
end


function Kronecker.getmatrices(K::KroneckerIdentityProduct)
    return (I(K.N), K.B)
end

function Kronecker.kronecker(A::Diagonal{Bool}, B::AbstractMatrix)
    # A is a diagonal matrix with boolean values
    # this is a special case where we can avoid forming the Kronecker product
    return KroneckerIdentityProduct(B, size(A, 1))
end

function LinearAlgebra.mul!(y::AbstractVector, K::KroneckerIdentityProduct, x::AbstractVector)
    kron_I_B_mv!(y, K.N, K.B, x)
    return y
end

function LinearAlgebra.mul!(Y::AbstractMatrix, K::KroneckerIdentityProduct, X::AbstractMatrix)
    kron_I_B_mm!(Y, K.N, K.B, X)
    return Y
end

function LinearAlgebra.mul!(Y::AbstractMatrix, K::KroneckerIdentityProduct, X::Adjoint{T, C}) where {T, C<:AbstractMatrix{T}}
    # make the adjoint explicit
    return mul!(Y, K, C(X))
end

function Base.:*(K::KroneckerIdentityProduct, v::AbstractVector)
    N = K.N
    m, q = size(K.B)
    y = similar(v, eltype(v), (m * N))
    mul!(y, K, v)
    return y
end
    
function Base.:*(K::KroneckerIdentityProduct, X::AbstractMatrix)
    N = K.N
    m, q = size(K.B)
    L = size(X, 2)
    Y = similar(X, eltype(X), (m * N, L))
    mul!(Y, K, X)
    return Y
end

function Base.:*(M::UpperTriangular{F, C}, K::KroneckerIdentityProduct) where {F, C <: AbstractMatrix{F}}
    return C(M) * K
end


# # mul!(C::Matrix{Float32}, A::UpperTriangular{Float32, CuArray{Float32, 2, CUDA.DeviceMemory}}, B::SpatiotemporalGPs.STGPKF.KroneckerIdentityProduct{Float32, Adjoint{Float32, CuArray{Float32, 2, CUDA.DeviceMemory}}})

function Base.:*(M::AbstractMatrix{F}, KT::KroneckerIdentityProduct{F, Adjoint{F, C}}) where {F, C <: AbstractMatrix{F}}

    # KT = (I ⊗ B)' = (I ⊗ B')
    # K = I ⊗ B

    # M * KT = M * (I ⊗ B')
    #        = ((I ⊗ B')' * M')'
    #        = (KT' * M')'
    #        = (K2 * M2)'

    # force the transpose to happen
    K2 = KroneckerIdentityProduct(C(KT.B'), KT.N)

    S = K2 * M'
    return S'

end

function Base.:*(M::UpperTriangular{F, C}, K::KroneckerIdentityProduct) where {F, C <: AbstractMatrix{F}}
    return C(M) * K
end 

function Base.:*(M::LowerTriangular{F, C}, K::KroneckerIdentityProduct) where {F, C <: AbstractMatrix{F}}
    return C(M) * K
end 

# function Base.:*(K::KroneckerIdentityProduct{F, C}, M::LinearAlgebra.AbstractTriangular{F}) where {F, C <: AbstractMatrix{F}}

#     # allocate the output
#     N = K.N
#     m, q = size(K.B)
#     L = size(M, 2)
#     @assert m == q # assuming B is square
#     @assert m * N == L # assuming output is a square matrix
#     @assert size(M, 1) == L 

#     Y = similar(M) # , F, (m * N, L))

#     # call our specialized function
#     return kron_I_B_mv!(Y, N, K.B, M) 
# end


# function Base.transpose(K::KroneckerIdentityProduct)
#     # transpose of (I(N) ⊗ B) is (I(N) ⊗ B')
#     return KroneckerIdentityProduct(transpose(K.B), K.N)
# end