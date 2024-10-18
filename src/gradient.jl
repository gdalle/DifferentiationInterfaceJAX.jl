struct AutoJAXGradientPrep{G,VG} <: DI.GradientPrep
    grad_jit::G
    value_and_grad_jit::VG
end

function DI.prepare_gradient(f, ::AutoJAX, x)
    grad_jit = jax[].jit(jax[].grad(f))
    value_and_grad_jit = jax[].jit(jax[].value_and_grad(f))
    return AutoJAXGradientPrep(grad_jit, value_and_grad_jit)
end

function DI.gradient(f, prep::AutoJAXGradientPrep, ::AutoJAX, x)
    (; grad_jit) = prep
    xp = pytensor(x)
    gp = grad_jit(xp)
    g = jltensor(gp)
    return g
end

function DI.value_and_gradient(f, prep::AutoJAXGradientPrep, ::AutoJAX, x)
    (; value_and_grad_jit) = prep
    xp = pytensor(x)
    yp, gp = value_and_grad_jit(xp)
    y = jltensor(yp)[]
    g = jltensor(gp)
    return y, g
end

function DI.gradient!(f, grad, prep::AutoJAXGradientPrep, backend::AutoJAX, x)
    return copyto!(grad, DI.gradient(f, prep, backend, x))
end

function DI.value_and_gradient!(f, grad, prep::AutoJAXGradientPrep, backend::AutoJAX, x)
    y, new_grad = DI.value_and_gradient(f, prep, backend, x)
    return y, copyto!(grad, new_grad)
end
