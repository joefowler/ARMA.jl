# Object RCPRoots is used to describe an array of complex numbers that
# could be the roots of a Real Coefficients Polynomial.
#
# This implies that for every non-real value p in the array, conj(p) is also.
#


"""
    struct RCPRoots

Describe an array of complex numbers such that for any value `z` in the set,
`conj(z)` is also. Because `z==conj(z)` for real `z`, real numbers automatically
satisfy it.
"""
struct RCPRoots{T<:AbstractVector} <: AbstractVector{T}
    "The array of complex values"
    z::T
    "How many values are complex in vector `z`"
    ncomplex::Int
end

function RCPRoots(z::AbstractVector{T}; angletol=1e-12, reltol=1e-12, abstol=1e-12) where T
    if T<:Real
        return RCPRoots(z, 0)
    end

    n = length(z)
    ϕ = angle.(z)
    isreal = abs.(sin.(ϕ)) .< angletol
    isreal[abs.(z) .< abstol] .= true
    # Values with complex phase w/i angle `angletol` of 0 or ±π are considered real.
    ncomplex = n-sum(isreal)
    if ncomplex%2 != 0
        throw(DomainError("RCPRoots called with an odd number of non-real values."))
    end
    if ncomplex == 0
        return RCPRoots(real(z), 0)
    end

    # Sort the complex-conjugate pairs. Each unsorted value is paired with the unsorted value nearest
    # to its complex conjugate. The two are then considered sorted.
    zsort = T[]
    zunsort = z[.~isreal]
    while length(zunsort) > 1
        c = zunsort[1]
        dist = abs.(c .- conj(zunsort[2:end]))
        mindist, idx = findmin(dist)
        idx = 1+idx  # because we omitted zunsort[1] from the test
        cc = zunsort[idx]  # best match to conj(c)
        if mindist > abstol || mindist/abs(c) > reltol
            throw(DomainError([c,cc], "RCPRoots called with closest pair [$c,$cc] not conjugate within tolerance"))
        end
        u = real(c+cc)/2
        v = abs(imag(c-cc)/2)
        append!(zsort, [u+v*1im, u-v*1im])
        deleteat!(zunsort, [1, idx])
    end

    # Real values go in the array last.
    append!(zsort, real(z[isreal]))
    RCPRoots(zsort, ncomplex)
end

# Make RCPRoots an iterable and indexable collection.
Base.getindex(rr::RCPRoots, idx) = rr.z[idx]
Base.firstindex(rr::RCPRoots) = 1
Base.lastindex(rr::RCPRoots) = length(rr.z)
Base.length(rr::RCPRoots) = length(rr.z)
Base.iterate(rr::RCPRoots, state=1) = state > length(rr.z) ? nothing : (rr.z[state], state+1)
Base.eltype(rr::RCPRoots) = eltype(rr.z)
Base.real(rr::RCPRoots) = real(rr.z)
Base.imag(rr::RCPRoots) = imag(rr.z)
function Base.size(rr::RCPRoots, dim=nothing)
    if dim === nothing
        return (length(rr.z),)
    elseif dim == 1
        return length(rr.z)
    elseif dim > 1
        return 1
    end
    throw(ErrorException("arraysize: dimension out of range"))
end

# """`ccpairs(rr::RCPRoots)` returns an array of 2-element arrays, the pairs of complex-conjugate values."""
# ccpairs(rr::RCPRoots) = [[i,i+1] for i=1:2:rr.ncomplex]
#
"`nreal(rr::RCPRoots)` returns the number of real values in `rr`."
nreal(rr::RCPRoots) = length(rr.z)-rr.ncomplex
"`ncomplex(rr::RCPRoots)` returns the number of complex values in `rr`."
ncomplex(rr::RCPRoots) = rr.ncomplex
"`realroots(rr::RCPRoots)` returns the real-valued elements of `rr` as an array of reals."
realroots(rr::RCPRoots) = real(rr.z[end+1-nreal(rr):end])
