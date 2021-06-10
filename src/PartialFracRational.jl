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

function roots(pfr::PartialFracRational; method=:Poly, nsteps=12)
    # nroots = pfr.b
    allowed = (:Poly, :Eig, :Both)
    if !(method in allowed)
        throw(ErrorException("method=$method, must be one of $allowed"))
    end
    if method == :Poly
        rough = roots_poly_method(pfr)
        return roots_improve(pfr, rough; nsteps)
    end

    if method == :Eig
        rough = roots_eigenvalue_method(pfr)
        return roots_improve(pfr, rough; nsteps)
    end

    rp = roots(pfr; method=:Poly, nsteps)
    re = roots(pfr; method=:Eig, nsteps)
    rp, re = promote(rp, re)
    r = eltype(rp)[]
    while length(rp) > 0
        idxp = argmin(abs2.(rp))
        idxe = argmin(abs2.(rp[idxp].-re))
        root_ideas = rp[idxp], re[idxe], 0.5*(re[idxp]+re[idxe])
        fvals = pfr.(root_ideas)
        idx = argmin(abs.(fvals))
        push!(r, root_ideas[idx])
        deleteat!(rp, idxp)
        deleteat!(re, idxe)
    end
    r
end

function roots_improve(pfr::PartialFracRational, rough::AbstractVector; nsteps=12)
    r = copy(rough)
    der = derivative(pfr)
    for i=1:length(r)
        x = xnew = r[i]
        newF = prevF = abs(pfr(x))
        # Try Newton steps, up to `nsteps`.
        for _ = 1:nsteps
            # Careful: very small _intended_ steps can actually yield no change at all
            # to floating-point precision. Check the actual step, not the intended one.
            intended_step = -pfr(x)/der(x)
            min_step = 3eps(abs(x))
            for _ = 1:9  # Shorten step by 3/5 up to 9 times (.6^9 = 0.01) before giving up
                abs(intended_step) ≤ min_step && break
                xnew = x+intended_step
                actual_step = xnew-x
                newF = abs(pfr(xnew))
                # @show newF, prevF, intended_step, actual_step, x
                newF < prevF && break
                # If the Newton step makes F worse
                intended_step *= 0.6
                # println("    shorten to $intended_step")
            end
            newF > prevF && break
            steptoosmall = abs(x-xnew) < min_step
            x, prevF = xnew, newF
            steptoosmall && break
        end
        # println()
        if eltype(r) <: Real
            r[i] = real(x)
        else
            r[i] = x
            # Sometimes better value with real(x). Check this.
            if abs(pfr(real(x))) < prevF
                r[i] = real(x)
            end
        end
    end
    r
end

function roots_poly_method(pfr::PartialFracRational)
    p = Polynomial(ChebyshevT(pfr.b))
    for j=1:pfr.n
        p *= Polynomial([-pfr.λ[j],1])
    end
    for i=1:pfr.n
        p1 = Polynomial(pfr.a[i])
        for j=1:pfr.n
            i==j && continue
            p1 *= Polynomial([-pfr.λ[j],1])
        end
        p += p1
    end
    roots(p)
end

"""
    partial_frac_decomp(num, poles::Vector)

Compute the partial fraction decomposition of num / Π_i=1:n (z-α[i])
as ∑ i=1:n r[i]/(z-α[i]). Return the coefficient vector `r`.

Uses the general relationship P(z)/Q(z) = ∑ P(α[i])/Q'(α[i]) * 1/(z-α[i])
"""
function partial_frac_decomp(num::Number, poles::AbstractVector)
    n = length(poles)
    T = promote_type(typeof(num), eltype(poles), Float64)
    r = zeros(T, n) .+ num

    for i=1:n
        for j=1:n
            j == i && continue
            r[i] /= (poles[i]-poles[j])
        end
    end
    r
end

function roots_eigenvalue_method(pfr::PartialFracRational; steptol=1e-14)
    # The pfr is a model of F(z)=N(z)/D(z)+R(z)
    # Now find a model for F(z)/R(z) = N(z)/ [D(z)R(z)] + 1 as a partial fraction
    # Once it's a full partial fraction, it will have new poles (the roots of R) but
    # the same roots.
    p, q = pfr.n, pfr.m
    if q<p
        return roots_pfrac0(pfr.a, pfr.λ)
    elseif q==p
        return roots_pfrac(pfr.a, pfr.λ, -pfr.b[end])
    end

    # q>p
    β = chebyshev_roots(real(pfr.b))
    if p==0
        return β
    end

    # q>p>0
    α = pfr.b[end]
    a = pfr.a / α
    bc = zeros(ComplexF64, q)
    for i=1:p
        poles = [pfr.λ[i], β...]
        pfc = partial_frac_decomp(a[i], poles)
        bc[i] = pfc[1]
        bc[end-(q-p)+1:end] .+= pfc[2:end]
    end
    roots_pfrac1(-bc, vcat(pfr.λ, β))
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
