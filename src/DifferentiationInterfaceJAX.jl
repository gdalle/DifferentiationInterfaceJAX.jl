module DifferentiationInterfaceJAX

using ADTypes
using DifferentiationInterface
import DifferentiationInterface as DI
using PythonCall

export AutoJAX

const jnp = Ref{Py}()
const jax = Ref{Py}()

struct AutoJAX <: AbstractADType end

struct AutoJAXGradientPrep{G} <: DI.GradientPrep
    grad_func::G
end

function DI.prepare_gradient(f, ::AutoJAX, x::X) where {X}
    grad_func = jax[].grad(f)
    return AutoJAXGradientPrep(grad_func)
end

function DI.gradient(f, prep::AutoJAXGradientPrep, ::AutoJAX, x)
    (; grad_func) = prep
    xnp = jnp[].array(x)
    gnp = grad_func(xnp)
    g = pyconvert(typeof(x), gnp)
    return g
end

function DI.value_and_gradient(f, prep::AutoJAXGradientPrep, ::AutoJAX, x)
    (; grad_func) = prep
    xnp = jnp[].array(x)
    ynp = f(xnp)
    gnp = grad_func(xnp)
    y = pyconvert(Array, ynp)[]
    g = pyconvert(typeof(x), gnp)
    return y, g
end

function DI.gradient!(f, grad, prep::AutoJAXGradientPrep, backend::AutoJAX, x)
    return copyto!(grad, DI.gradient(f, prep, backend, x))
end

function DI.value_and_gradient!(f, grad, prep::AutoJAXGradientPrep, backend::AutoJAX, x)
    y, new_grad = DI.value_and_gradient(f, prep, backend, x)
    return y, copyto!(grad, new_grad)
end

function __init__()
    jax[] = pyimport("jax")
    jnp[] = pyimport("jax.numpy")
    return nothing
end

end
