"""
    struct PairedPartialFracRational

Represents an order `(q,p)` rational function `f` with real coefficients in "paired partial
fraction form." Useful for parameter optimization, because it's still in the well-behaved partial
fraction form, but its coefficients are all real.

"""
struct PairedPartialFracRational{T <: Real}
    u::Vector{T}

    q::Int  # degree of the numerator polynomial
    p::Int  # degree of the denominator polynomial
    m::Int  # number of residues, ≤p and ≤q+1
end

Base.:*(scale::Real, ppfr::PairedPartialFracRational) = Base.:*(ppfr, scale)
function Base.:*(ppfr::PairedPartialFracRational, scale::Real)
    u = copy(ppfr.u)
    u[1:ppfr.q+1] *= scale
    PairedPartialFracRational(u, ppfr.q, ppfr.p, ppfr.m)
end

# An alias: calling a `PairedPartialFracRational` is equivalent to calling `ppfrat_eval` on it.
(ppfr::PairedPartialFracRational)(z) = ppfrat_eval(z, ppfr)

"""
    ppfrat_eval(z, ppfr::PairedPartialFracRational)

Evaluate the rational function `ppfr` at `z`, which may be a number or an AbstractArray.
Returns as number or array of the same form and shape as `z` (but promoted to floating point, possibly complex).
"""
function ppfrat_eval(z::AbstractVector, ppfr::PairedPartialFracRational)
    # Make sure to end up with at least a Float, but complex if any of {z, ppfr.u} are complex.
    T = promote_type(eltype(z), eltype(ppfr.u), Float64)
    z = convert(Vector{T}, z)

    unum = ppfr.u[1:ppfr.m]
    remainder = Polynomial(ppfr.u[ppfr.m+1:ppfr.q+1])
    f = remainder.(z)
    udenom = ppfr.u[ppfr.q+2:end]
    @assert length(udenom) == ppfr.p

    m_pairs = div(ppfr.m, 2)
    for i=1:m_pairs
        f .+= (unum[2i-1]*z .+ unum[2i]) ./(z.^2 .- udenom[2i-1]*z .+ udenom[2i])
    end
    if ppfr.m%2 == 1
        f .+= unum[ppfr.m] ./ (z .- udenom[ppfr.m])
    end

    if ppfr.p > ppfr.m
        e_pairs = div(ppfr.p-ppfr.m, 2)
        udenom = udenom[ppfr.m+1:end]
        for i=1:e_pairs
            f ./= z.^2 .- udenom[2i-1]*z .+ udenom[2i]
        end
        if (ppfr.p-ppfr.m)%2 == 1
            f ./= z .- udenom[end]
        end
    end
    f
end
ppfrat_eval(z::Number, ppfr::PairedPartialFracRational) = ppfrat_eval([z], ppfr)[1]
