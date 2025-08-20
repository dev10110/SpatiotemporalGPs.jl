GeneralizedKroneckerProduct = Kronecker.GeneralizedKroneckerProduct

# provide a way to allocate cuda arrays if the v is a cuda vector
function Base.:*(K::GeneralizedKroneckerProduct, v::CuVector)
    return mul!(CuVector{promote_type(eltype(v), eltype(K))}(undef, first(size(K))), K, v)
end

# provide a way to allocate cuda arrays if the M is a cuda matrix
function Base.:*(K::GeneralizedKroneckerProduct, M::CuMatrix)
    return mul!(CuMatrix{promote_type(eltype(M), eltype(K))}(undef, size(K, 1), size(M, 2)), K, M)
end


function Base.:*(v::CuMatrix, K::GeneralizedKroneckerProduct)
    out = CuMatrix{promote_type(eltype(v), eltype(K))}(undef, last(size(K)), first(size(v)))
    # need to use copy instead of collect to keep the CuArray type
    return transpose(mul!(out, transpose(K), copy(transpose(v))))
end