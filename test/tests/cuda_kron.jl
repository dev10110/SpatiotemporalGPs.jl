

using Test
using LinearAlgebra
using StaticArrays
using CUDA
using SpatiotemporalGPs
using Kronecker

println("CUDA Functional: $(CUDA.functional())")


@testset "Kron test" begin

    n = 3
    N = 10
    A = randn(n, n)
    b = randn(n * N)
    In = I(N)
    K = In ⊗ A

    @assert typeof(K) <: STGPKF.KroneckerIdentityProduct

    # check against julia's kron
    @test K * b ≈ kron(In, A) * b


    # now check matrix multiplication
    B = randn(n * N, N)
    @test K * B ≈ kron(In, A) * B

    # test adjoint
    Kt = K'
    @test Kt * b ≈ kron(In, A') * b
    @test Kt * B ≈ kron(In, A') * B


end


@testset "CUDA Kron test" begin

    # same as above but for CUDA

    n = 3
    N = 10
    A = randn(n, n)
    cu_A = cu(A)
    In = I(N)

    b = randn(n * N)
    cu_b = cu(b)

    # create the cuda version of the product
    K = In ⊗ cu_A

    @test typeof(K) <: STGPKF.KroneckerIdentityProduct

    # check against julia's kron
    @test collect(K * cu_b) ≈ collect(kron(In, A) * b)

    # now check matrix multiplication
    B = randn(n * N, N)
    cu_B = cu(B)
    @test collect(K * cu_B) ≈ collect(kron(In, A) * B)

end

# test adjoint products
@testset "CUDA Kron adjoint test" begin

    n = 3
    N = 10
    A = randn(n, n)
    cu_A = cu(A)
    In = I(N)

    # create the cuda version of the product
    K = In ⊗ cu_A

    # test adjoint
    Kt = K'

    # test adjoint product
    M = randn(n * N, n * N)
    cu_M = cu(M)
    @test collect(Kt * cu_M) ≈ collect(kron(In, A') * M)
    @test collect(cu_M * Kt) ≈ collect(M * kron(In, A'))


end
