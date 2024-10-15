using DifferentiationInterface
using DifferentiationInterfaceJAX
using PythonCall
using Test

jnp = pyimport("jax.numpy")
jax = pyimport("jax")

backend = AutoJAX()

f(x) = jnp.sum(jnp.square(x))
f_jit = jax.jit(f)

x = float.(1:3)
grad = similar(x)

@testset "Without JIT" begin
    @test gradient(f, backend, x) == 2x
    @test gradient!(f, grad, backend, x) == 2x
    @test value_and_gradient(f, backend, x) == (sum(abs2, x), 2x)
    @test value_and_gradient!(f, grad, backend, x) == (sum(abs2, x), 2x)
end

@testset "With JIT" begin
    @test gradient(f_jit, backend, x) == 2x
    @test gradient!(f_jit, grad, backend, x) == 2x
    @test value_and_gradient(f_jit, backend, x) == (sum(abs2, x), 2x)
    @test value_and_gradient!(f_jit, grad, backend, x) == (sum(abs2, x), 2x)
end
