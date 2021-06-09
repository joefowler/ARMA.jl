"""
    struct PartialFracRational

Represent an order `(m,n)` rational function in partial fraction form. That is, as

    f(z) = Σ i=1:n a[i]/(z-λ[i]) + Σ i=0:(m-n) b[i+1]*T_i(zs)

where the first sum is the partial fraction expansion, and the second is the polynomial remainder, expanded
in terms of Chebyshev polynomials. The remainder argument `zs` is equal to `z` by default but can be changed
to an affine transformation of `z` instead.
- `λ[1:n]` are the simple poles of f
- `a[1:n]` are the residues at those poles
- `b[1:m-n+1]` are the coefficients of the Chebyshev polynomial expansion
- `T_i` is the Chebyshev polynomial of degree i

This object cannot accomodate poles with multiplicity > 1, unfortunately. In practice, however,
the effect of a multiple pole can be approximated as closely as needed by two nearby poles.

# Constructors

    PartialFracRational(λ, a, b; <keyword arguments>)
    PartialFracRational(λ, a; <keyword arguments>)
    PartialFracRational(b; <keyword arguments>)

# Arguments
- `λ`: vector of poles
- `a`: vector of residues (must be same length as `λ`)
- `b`: vector of coefficients of Chebyshev polynominals in the remainder. If `b` is omitted, the remainder
  is assumed to be zero. If both `λ` and `a` are omitted, no partial fractions are assumed. They cannot
  all be omitted.

# Keyword arguments
- `polyMin::Number=-1`
- `polyMax::Number=+1`: If the usual domain of the Chebyshev polynomials [-1,+1] is
    not appropriate for this problem, these values can be changed so that `[polyMin,polyMax]` is affine
    tranformed to [-1,+1] before computing the Chebyshev terms. These values affect _only_ the remainder terms,
    not the computation of the partial fraction terms. These values have no effect when
    `length(b)` ≤ 1 (i.e., the remainder polynominal is a constant).

"""
struct PartialFracRational{T <: AbstractVector, U <: Number}
    λ::T  # poles
    a::T  # residues
    b::T  # coefficients of the remainder polynomial, a Chebyshev series

    m::Int  # degree of the numerator polynomial
    n::Int  # degree of the denominator polynomial
    polyMin::U
    polyMax::U
end

# Constructors
PartialFracRational(b::AbstractVector{U};polyMin::Number=-1, polyMax::Number=+1) where {U} =
    PartialFracRational(Float64[], Float64[], b; polyMin, polyMax)

function PartialFracRational(λ::AbstractVector{S}, a::AbstractVector{T}, b::AbstractVector{U}=[];
    polyMin::Number=-1, polyMax::Number=+1) where {S, T, U}
    if b == []
        b = eltype(a)[]
    end
    λ, a, b = promote(float(λ), float(a), float(b))
    polyMin, polyMax = promote(polyMin, polyMax)

    n = length(λ)
    if n != length(a)
        throw(DimensionMismatch("length(λ) $(length(λ)) != length(a) $(length(a))"))
    end
    m = length(b)-1+n
    @assert m ≥ n-1
    @assert polyMax != polyMin
    PartialFracRational(λ, a, b, m, n, polyMin, polyMax)
end

Base.:*(scale::Number, pfr::PartialFracRational) = Base.:*(pfr, scale)
Base.:*(pfr::PartialFracRational, scale::Number) = PartialFracRational(pfr.λ, pfr.a*scale, pfr.b*scale; pfr.polyMin, pfr.polyMax)


(pfr::PartialFracRational)(z) = pfrat_eval(z, pfr)

"""
    pfrat_eval(z, pfr::PartialFracRational)

Evaluate the rational function `pfr` at `z`, which may be a number or an AbstractArray.
Returns as number or array of the same form and shape as `z` (but promoted to floating point, possibly complex).
"""
function pfrat_eval(z, pfr::PartialFracRational)
    arg_isscalar = size(z) == ()
    zvec = arg_isscalar ? [z] : vec(z)
    zscaled = 2(zvec .- pfr.polyMin)/(pfr.polyMax-pfr.polyMin) .- 1
    # Make sure to end up with at least a Float, but complex if any of {z, pfr.a, or pfr.polyMin} are complex.
    T = promote_type(eltype(z), eltype(pfr.a), typeof(pfr.polyMin), Float64)
    remainder = ChebyshevT(pfr.b)
    f = evalpoly.(zscaled, remainder, false)
    for i=1:pfr.n
        f .+= pfr.a[i]./(zvec.-pfr.λ[i])
    end
    f = arg_isscalar ? f[1] : reshape(f, size(z))
end


"""
    derivative(pfr::PartialFracRational, order::Int=1)

Returns a function that evaluates the derivative of `pfr`.
"""
function derivative(pfr::PartialFracRational, order::Int=1)
    order <= 0 && return pfr
    # Scale = (-1)^order * factorial(order)
    scale = -1.0
    for i=2:order
        scale *= -i
    end
    function der(x::Number)
        f0 = 0.0im
        for (ρ, λ) in zip(pfr.a, pfr.λ)
            f0 += scale*ρ/(x-λ)^(order+1)
        end
        f0 + evalpoly(complex(x), derivative(ChebyshevT(pfr.b), order), false)
    end
    der
end

"""
    legendre_companion(c)

Return the scaled companion matrix for a Legendre series with coefficients `c`.
Copied from numpy/polynomial/legendre.py
"""
function legendre_companion(c::AbstractVector)
    if length(c) < 2
        throw(ErrorException("Series must have at least 2 terms (degree ≥ 1)."))
    elseif length(c)==2
        return [[-c[1]/c[2]]]
    end
    n = length(c)-1
    scale = collect(1:2:2n-1).^-0.5
    top = collect(1:n-1).*scale[1:n-1].*scale[2:end]
    # mat = zeros(eltype(c), n, n)
    mat = diagm(1=>top, 0=>zeros(eltype(c), n), -1=>top)
    mat[:, end] .-= (c[1:end-1]/c[end]).*(scale/scale[end])*(n/(2n-1))
    mat
end

"""
    legendre_roots(c)

Return the roots of a Legendre series with coefficients `c`.
Copied from numpy/polynomial/legendre.py
"""
function legendre_roots(c::AbstractVector)
    if length(c) < 2
        return eltype(c)[]
    elseif length(c)==2
        return [-c[1]/c[2]]
    end
    # Rotated companion matrix reduces error, supposedly.
    m = legendre_companion(c)[end:-1:1, end:-1:1]
    eigvals(m)
end


"""
    chebyshev_companion(c)

Return the scaled companion matrix for a Chebyshev series with coefficients `c`.
Copied from numpy/polynomial/chebyshev.py
"""
function chebyshev_companion(c::AbstractVector)
    if length(c) < 2
        throw(ErrorException("Series must have at least 2 terms (degree ≥ 1)."))
    elseif length(c)==2
        return [[-c[1]/c[2]]]
    end
    n = length(c)-1
    scale = fill(sqrt(0.5), n)
    scale[1] = 1.0

    top = scale[1:n-1]*sqrt(0.5)
    # mat = zeros(eltype(c), n, n)
    mat = diagm(1=>top, 0=>zeros(eltype(c), n), -1=>top)
    mat[:, end] .-= (c[1:end-1]/c[end]).*(scale/scale[end])*0.5
    mat
end

"""
    chebyshev_roots(c)

Return the roots of a Chebyshev series with coefficients `c`.
Copied from numpy/polynomial/chebyshev.py
"""
function chebyshev_roots(c::AbstractVector)
    if length(c) < 2
        return eltype(c)[]
    elseif length(c)==2
        return [-c[1]/c[2]]
    end
    # Rotated companion matrix reduces error, supposedly.
    m = chebyshev_companion(c)[end:-1:1, end:-1:1]
    eigvals(m)
end
