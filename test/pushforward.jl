using DifferentiationInterface
using DifferentiationInterfaceJAX
using PythonCall
using Test

jnp = pyimport("jax.numpy")
jax = pyimport("jax")

backend = AutoJAX()

f(x) = jnp.square(x)

x = float.(1:3)
tx = (float.(4:6),)
ty = map(similar, tx)

@testset "Without JIT" begin
    @test pushforward(f, backend, x, tx) == (2x .* only(tx),)
    @test pushforward!(f, ty, backend, x, tx) == (2x .* only(tx),)
    @test value_and_pushforward(f, backend, x, tx) == (abs2.(x), (2x .* only(tx),))
    @test value_and_pushforward!(f, ty, backend, x, tx) == (abs2.(x), (2x .* only(tx),))
end
