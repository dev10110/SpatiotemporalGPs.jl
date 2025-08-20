
# provide a overload for A * B' that works with CUDA
function LinearAlgebra.mul!(C::TC,
    A::UpperTriangular{F, TC},
    B::Adjoint{F, TC}) where {F, TC <: CuMatrix{F}}

    # force the copy when running with adjoint
    return mul!(C, A, copy(B) )
end



function LinearAlgebra.mul!(C::TC,  A::Matrix{F}, B::Symmetric{F, TC}) where {F, TC <: CuArray{F}}
    # SO SHITTY!
    cA = CuArray{F}(A)
    n = size(B, 1) 
    cB = CuArray{F}(I(n)) * B
    return mul!(C, cA, cB)
end


