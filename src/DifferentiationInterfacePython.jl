module DifferentiationInterfacePython

using ADTypes
using DifferentiationInterface
import DifferentiationInterface as DI
using PythonCall

export AutoAutoGrad

const np = Ref{Py}()
const ag = Ref{Py}()

struct AutoAutoGrad <: AbstractADType end

struct AutoAutoGradGradientPrep{G} <: DI.GradientPrep
    grad_func::G
end

function DI.prepare_gradient(f, ::AutoAutoGrad, x)
    grad_func = ag[].grad(pyfunc(f))
    return AutoAutoGradGradientPrep(grad_func)
end

function DI.gradient(f, prep::AutoAutoGradGradientPrep, ::AutoAutoGrad, x)
    (; grad_func) = prep
    xnp = np[].array(x)
    gnp = grad_func(xnp)
    g = pyconvert(typeof(x), gnp)
    return g
end

function __init__()
    ag[] = pyimport("autograd")
    np[] = pyimport("autograd.numpy")
end

end
