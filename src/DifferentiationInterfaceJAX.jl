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

function DI.prepare_gradient(f, ::AutoJAX, x)
    grad_func = jax[].grad(pyfunc(f))
    return AutoJAXGradientPrep(grad_func)
end

function DI.gradient(f, prep::AutoJAXGradientPrep, ::AutoJAX, x)
    (; grad_func) = prep
    xnp = jnp[].array(x)
    gnp = grad_func(xnp)
    g = pyconvert(typeof(x), gnp)
    return g
end

function __init__()
    jax[] = pyimport("jax")
    jnp[] = pyimport("jax.numpy")
    return nothing
end

end
