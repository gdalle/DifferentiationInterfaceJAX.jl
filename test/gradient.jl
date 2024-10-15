using DifferentiationInterface
using DifferentiationInterfaceJAX
using PythonCall
using Test

jnp = pyimport("jax.numpy")

backend = AutoJAX()

f(x) = jnp.sum(jnp.square(x))

x = float.(1:3)
grad = similar(x)

@test gradient(f, backend, x) == [2, 4, 6]
@test gradient!(f, grad, backend, x) == [2, 4, 6]
@test value_and_gradient(f, backend, x) == (14, [2, 4, 6])
@test value_and_gradient!(f, grad, backend, x) == (14, [2, 4, 6])
