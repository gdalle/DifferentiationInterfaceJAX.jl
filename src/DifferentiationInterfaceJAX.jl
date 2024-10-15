module DifferentiationInterfaceJAX

using ADTypes
using DifferentiationInterface
import DifferentiationInterface as DI
using PythonCall

export AutoJAX

const jnp = Ref{Py}()
const jax = Ref{Py}()

struct AutoJAX <: AbstractADType end

ADTypes.mode(::AutoJAX) = ADTypes.ForwardOrReverseMode()

## Pushforward

function DI.prepare_pushforward(f, ::AutoJAX, x, tx::NTuple)
    return DI.NoPushforwardPrep()
end

function DI.pushforward(f, ::DI.NoPushforwardPrep, ::AutoJAX, x, tx::NTuple)
    xnp = jnp[].array(x)
    ty = map(tx) do dx
        dxnp = jnp[].array(dx)
        _, dynp = jax[].jvp(f, (xnp,), (dxnp,))
        dy = pyconvert(Array, dynp)
    end
    return ty
end

function DI.value_and_pushforward(f, ::DI.NoPushforwardPrep, ::AutoJAX, x, tx::NTuple)
    xnp = jnp[].array(x)
    ys_and_ty = map(tx) do dx
        dxnp = jnp[].array(dx)
        ynp, dynp = jax[].jvp(f, (xnp,), (dxnp,))
        y = pyconvert(Array, ynp)
        dy = pyconvert(Array, dynp)
        y, dy
    end
    y = first(ys_and_ty[1])
    ty = last.(ys_and_ty)
    return y, ty
end

function DI.pushforward!(
    f, ty::NTuple, prep::DI.NoPushforwardPrep, backend::AutoJAX, x, tx::NTuple
)
    new_ty = DI.pushforward(f, prep, backend, x, tx)
    foreach(copyto!, ty, new_ty)
    return ty
end

function DI.value_and_pushforward!(
    f, ty::NTuple, prep::DI.NoPushforwardPrep, backend::AutoJAX, x, tx::NTuple
)
    y, new_ty = DI.value_and_pushforward(f, prep, backend, x, tx)
    foreach(copyto!, ty, new_ty)
    return y, ty
end

## Gradient

struct AutoJAXGradientPrep{G} <: DI.GradientPrep
    grad_func::G
end

function DI.prepare_gradient(f, ::AutoJAX, x)
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
