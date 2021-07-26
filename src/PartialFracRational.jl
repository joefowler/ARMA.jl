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
            f0 += ρ/(x-λ)^(order+1)
        end
        scale*f0 + evalpoly(complex(x), derivative(ChebyshevT(pfr.b), order), false)
    end
    der
end

"""
    roots(pfr::PartialFracRational; method=:Eig, nsteps=25)

Compute the roots of partial-fraction rational function `pfr`.

Allowed methods are `(:Eig, :Poly, :Both)`. The default, `:Eig`, will use the
eignvalue method of Knockaert (for example). `:Poly`, will compute the equivalent
numerator polynomial by its coefficients and use `Polynomials.roots()` to find the
roots. Either method will then:

1.  Improve a root by up to `nsteps` steps of Laguerre's Method. The method is used not on `pfr`
    but on the product of `pfr` and its denominator factors. This procedure yields the numerator polynomial
    implictly, without having to compute it directly.
2.  As each root is improved, it is added to the set of known roots, so that the numerator is used
    with _zero suppression_. That is, roots are found not for the numerator, but for the numerator
    divided by the product of (x-ξ) for all roots {ξ} found so far. This zero suppression allows
    us to find multiple roots, or near-identical roots.

If method `:Both` is chosen, then both are tried, the results from each are paired up (each root to
the nearest from the other method). From each pair, the two values and the mean of the two are all
tested in function `pfr`. Whichever of the three has the smallest absolute result is chosen as a root.
"""
function roots(pfr::PartialFracRational; method=:Eig, nsteps=65)
    allowed = (:Eig, :Poly, :Both)
    if !(method in allowed)
        throw(ErrorException("method=$method, must be one of $allowed"))
    end
    if method == :Poly
        rough = roots_poly_method(pfr)
        return improve_roots_laguerre(pfr, rough; nsteps)
    end

    if method == :Eig
        rough = roots_eigenvalue_method(pfr)
        return improve_roots_laguerre(pfr, rough; nsteps)
    end

    # Here use :Both. Try both methods, match up roots, and use whichever is best.
    rp = roots(pfr; method=:Poly, nsteps)
    re = roots(pfr; method=:Eig, nsteps)
    rp, re = promote(rp, re)
    r = eltype(rp)[]
    # Go through roots in order of rp from lowest absolute value.
    # Match each root in re that is closest to the one from rp.
    # Now compare the two roots and their mean. Choose the best of the three.
    # (Here, "best" means smallest absolute value of `pfr(x)`.)
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

"""
    improve_roots_laguerre(pfr::PartialFracRational, rough::AbstractVector; nsteps=12)

Return an improved estimate of the roots of rational function `pfr`, given the initial
estimates `rough`. Do this by taking `nsteps` steps of Laguerre's method. There is no
guarantee that a rough estimate will converge to the nearest actual root. However,
the use of zero suppression and the near-global convergence of Laguerre's method to
some root means that we don't really have to worry about this.

At most `nsteps` will be taken, but iteration will stop early if the step
is close to the machine precision.
"""
function improve_roots_laguerre(pfr::PartialFracRational, rough::AbstractVector; nsteps=75)
    knownroots = ComplexF64[]
    f1 = derivative(pfr,1)
    f2 = derivative(pfr,2)

    # Functions to compute the numerator and denominator, and also r,
    # where r(x) ≡ denom'(x)/denom(x), and its derivative r1 ≡ dr/dx.
    denom(x) = prod([x-r for r in pfr.λ])
    numer(x) = pfr(x)*denom(x)
    r(x) = sum([1.0/(x-ri) for ri in pfr.λ])
    r1(x) = -sum([1.0/(x-ri)^2 for ri in pfr.λ])
    # @show rough
    # @show abs.(pfr(rough))

    for i=1:length(rough)
        n = pfr.m-length(knownroots)

        # For zero suppression, we need Q(x), the product of all so-far known
        # factors in the numerator, and R(x) ≡ Q'(x)/Q(x) and R'(x).
        Q(x) = prod([(x-kr) for kr in knownroots])
        R(x) = sum([1.0/(x-kr) for kr in knownroots])
        R1(x) = -sum([1.0/(x-kr)^2 for kr in knownroots])

        x = rough[i]
        bestf, bestx = abs(pfr(x)), x

        # Try steps of Laguerre's method, up to `nsteps` or until steps are too small.
        for iter = 1:nsteps
            fratio = f1(x)/pfr(x)
            G = fratio+r(x)-R(x)
            H = fratio^2-f2(x)/pfr(x)-r1(x)+R1(x)
            rt = sqrt(complex(n-1.0)*(n*H-G^2))
            dd = G+rt
            if abs(G-rt) > abs(dd)
                dd = G-rt
            end
            newx = x-n/dd
            # To break the rare case of limit cycles, try ~half-sized steps 1 time out of 10.
            if iter%10 == 0
                fraction = .5*rand()*.25
                newx = x-fraction*n/dd
            end
            absf = abs(pfr(newx))
            if absf < bestf
                bestf, bestx = absf, newx
            end
            actual_step = abs(x-newx)
            # @show iter, newx, actual_step, pfr(newx)
            real(actual_step) ≤ 2eps(real(x)) && imag(actual_step) ≤ 2eps(imag(x)) && break
            x = newx
        end
        # Sometimes better value with real(x). Check this.
        if abs(pfr(real(bestx))) ≤ bestf
            bestx = real(bestx)
        end
        push!(knownroots, bestx)
    end
    knownroots
end

"""
    roots_poly_method(pfr::PartialFracRational)

Estimate the roots of rational function `pfr` by the method of computing the effective
numerator polynomial (by coefficients) and finding its roots. For polynomials of high
degree (`pfr.m` ≳ 5-10), the computation by coefficients can lead to poor conditioning
and dubious results.
"""
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

"""
    roots_eigenvalue_method(pfr::PartialFracRational)

Estimate the roots of rational function `pfr` by the method of eigenvalues.
"""
function roots_eigenvalue_method(pfr::PartialFracRational)
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
