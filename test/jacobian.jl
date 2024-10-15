using DifferentiationInterface
using DifferentiationInterfaceJAX
using LinearAlgebra
using PythonCall
using SparseArrays
using SparseMatrixColorings
using Test

jnp = pyimport("jax.numpy")
jax = pyimport("jax")

backend = AutoJAX()
sparse_backend = AutoSparse(
    backend;
    sparsity_detector=DenseSparsityDetector(backend; atol=1e-3),
    coloring_algorithm=GreedyColoringAlgorithm(),
)

f(x) = jnp.square(x)
f(x::AbstractArray) = pyconvert(Array, f(jnp.array(x)))

x = float.(1:3)
J = similar(x, length(x), length(x))

@testset "Dense" begin
    @test jacobian(f, backend, x) == Diagonal(2x)
    @test jacobian!(f, J, backend, x) == Diagonal(2x)
    @test value_and_jacobian(f, backend, x) == (abs2.(x), Diagonal(2x))
    @test value_and_jacobian!(f, J, backend, x) == (abs2.(x), Diagonal(2x))
end

@testset "Sparse" begin
    @test jacobian(f, sparse_backend, x) isa SparseMatrixCSC
    @test jacobian(f, sparse_backend, x) == Diagonal(2x)
    @test jacobian!(f, J, sparse_backend, x) == Diagonal(2x)
    @test value_and_jacobian(f, sparse_backend, x) == (abs2.(x), Diagonal(2x))
    @test value_and_jacobian!(f, J, sparse_backend, x) == (abs2.(x), Diagonal(2x))
end
