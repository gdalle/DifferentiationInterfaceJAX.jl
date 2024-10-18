module DifferentiationInterfaceJAX

using ADTypes: ADTypes
import DifferentiationInterface as DI
using DLPack: from_dlpack, share
using PythonCall: Py, pyconvert, pyimport

const jnp = Ref{Py}()
const jax = Ref{Py}()

function __init__()
    jax[] = pyimport("jax")
    jnp[] = pyimport("jax.numpy")
    return nothing
end

pytensor(xj) = share(xj, jax[].dlpack.from_dlpack)
jltensor(xp) = from_dlpack(xp)

struct AutoJAX <: ADTypes.AbstractADType end

ADTypes.mode(::AutoJAX) = ADTypes.ForwardOrReverseMode()
DI.inplace_support(::AutoJAX) = DI.InPlaceNotSupported()

include("gradient.jl")
include("pushforward.jl")

export AutoJAX
export pytensor, jltensor

end
