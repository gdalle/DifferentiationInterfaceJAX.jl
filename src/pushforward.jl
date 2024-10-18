
function DI.prepare_pushforward(f, ::AutoJAX, x, tx::NTuple)
    return DI.NoPushforwardPrep()
end

function DI.pushforward(f, ::DI.NoPushforwardPrep, ::AutoJAX, x, tx::NTuple)
    xp = pytensor(x)
    ty = map(tx) do dx
        dxp = pytensor(dx)
        _, dyp = jax[].jvp(f, (xp,), (dxp,))
        dy = jltensor(dyp)
    end
    return ty
end

function DI.value_and_pushforward(f, ::DI.NoPushforwardPrep, ::AutoJAX, x, tx::NTuple)
    xp = pytensor(x)
    ys_and_ty = map(tx) do dx
        dxp = pytensor(dx)
        yp, dyp = jax[].jvp(f, (xp,), (dxp,))
        y = jltensor(yp)
        dy = jltensor(dyp)
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
