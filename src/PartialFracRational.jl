"""
    struct PartialFracRational

Represents an order `(q,p)` rational function `f` in partial fraction form. That is, as

    f(z) = [Σ i=1:m a[i]/(z-λ[i]) + Σ j=0:(q-m) b[i]*z^j] * [ ∏ i=m+1:p 1/(z-λ[i])]
    f(z) = [P(z) + R(z)]*E(z)

where the first sum `P` is the partial fraction expansion, and the second `R` is the polynomial
remainder.  The final product `E` contains extra poles when they outnumber
the roots by at least one (`p>q+1`).

- `(q,p)` are the degree of the numerator and numerator, respectively.
- `m` is the number of residues, either `min(p,q+1)`, or one less than this (in the special case that
  `p` and `q` are both even and `q<p`).
- `λ[1:p]` are the simple poles of `f`.
- `a[1:m]` are the residues at the first `m≤p` poles.
- `b[1:q-m+1]` are the coefficients of the polynomial expansion of the remainder term.

Because of how `m` is constrained, exactly one of the following is true:

1. The product `E(z)` equals 1 (happens when `m=p`).
2. The second sum `R(z)` equals 0 (when `m=q+1`).
3. The second sum `R(z)` equals a non-zero constant (when `m=q`).

This object cannot accomodate poles with multiplicity > 1, unfortunately. In practice, however,
the effect of a multiple pole can be approximated as closely as needed by two nearby poles.

# Constructors

    PartialFracRational(λ, a, b)
    PartialFracRational(λ, a)
    PartialFracRational(b)

# Arguments
- `λ`: vector of poles
- `a`: vector of residues (must be no longer than `λ`)
- `b`: vector of coefficients of monominals in the remainder, or a constant to represent
  a constant remainder. If `b` is omitted, the remainder is assumed to be zero. If both `λ` and `a`
  are omitted, no partial fractions are assumed. They cannot all be omitted.
"""
struct PartialFracRational{T <: AbstractVector}
    λ::T  # poles
    a::T  # residues
    b::T  # coefficients of the remainder polynomial

    q::Int  # degree of the numerator polynomial
    p::Int  # degree of the denominator polynomial
    m::Int  # number of residues, ≤p and ≤q+1
end

# Constructors
PartialFracRational(b::AbstractVector{U}) where {U} =
    PartialFracRational(Float64[], Float64[], b)
PartialFracRational(b::Real) = PartialFracRational(Float64[], Float64[], [b])

PartialFracRational(λ::RCPRoots, a::AbstractVector{T}, b::AbstractVector{U}=[]) where {T, U} = PartialFracRational(λ.z, a, b)
PartialFracRational(λ::RCPRoots, a::AbstractVector{T}, b::Number) where {T} = PartialFracRational(λ.z, a, [b])
PartialFracRational(λ::AbstractVector{S}, a::AbstractVector{T}, b::Number) where {S, T} = PartialFracRational(λ, a, [b])

function PartialFracRational(λ::AbstractVector{S}, a::AbstractVector{T}, b::AbstractVector{U}=[]) where {S, T, U}
    if b == []
        b = eltype(a)[]
    end
    λ, a, b = promote(float(λ), float(a), float(b))

    p = length(λ)
    m = length(a)
    if m > p
        throw(DimensionMismatch("length(a) $(m) > length(λ) $(p)"))
    end
    q = length(b)-1+m
    if m > q+1
        throw(DimensionMismatch("length(a) $(m) > length(a)+length(b)+1 $(q+1)"))
    end
    PartialFracRational(λ, a, b, q, p, m)
end

Base.:*(scale::Number, pfr::PartialFracRational) = Base.:*(pfr, scale)
Base.:*(pfr::PartialFracRational, scale::Number) = PartialFracRational(pfr.λ, pfr.a*scale, pfr.b*scale)

# An alias: calling a `PartialFracRational` is equivalent to calling `pfrat_eval` on it.
(pfr::PartialFracRational)(z) = pfrat_eval(z, pfr)

"""
    pfrat_eval(z, pfr::PartialFracRational)

Evaluate the rational function `pfr` at `z`, which may be a number or an AbstractArray.
Returns as number or array of the same form and shape as `z` (but promoted to floating point, possibly complex).
"""
function pfrat_eval(z::AbstractArray, pfr::PartialFracRational)
    # Make sure to end up with at least a Float, but complex if any of {z, pfr.a} are complex.
    T = promote_type(eltype(z), eltype(pfr.a), Float64)
    z = convert(Vector{T}, z)
    remainder = Polynomial(pfr.b)
    f = remainder.(z)
    for i=1:pfr.m
        f .+= pfr.a[i]./(z.-pfr.λ[i])
    end
    for i=1+pfr.m:pfr.p
        f ./= (z.-pfr.λ[i])
    end
    f
end
pfrat_eval(z::Number, pfr::PartialFracRational) = pfrat_eval([z], pfr)[1]

"""
    derivative(pfr::PartialFracRational, order::Int=1)

Returns a function that evaluates the derivative of `pfr`.
"""
function derivative(pfr::PartialFracRational, order::Int=1)
    order <= 0 && return pfr

    # A PFR with p>m will be hard to differentiate analytically. Thus,
    # "upgrade" it to order (q,p) with q=p before differentiating.
    if pfr.p > pfr.m
        pfr = upgrade(pfr)
    end
    @assert pfr.p ≤ pfr.m

    # Scale = (-1)^order * factorial(order)
    scale = -1.0
    for i=2:order
        scale *= -i
    end

    function der(x::Number)
        f0 = 0.0im
        for (ρ, λ) in zip(pfr.a, pfr.λ[1:pfr.m])
            f0 += ρ/(x-λ)^(order+1)
        end
        scale*f0 + evalpoly(complex(x), derivative(Polynomial(pfr.b), order))
    end
    der
end

import Polynomials.roots, Polynomials.derivative
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

    # A PFR with p>m will be hard to differentiate analytically. Thus,
    # change it to order (q,m) (where m=q+1 or m=q) by dropping those final (p-m) factors before
    # differentiating. They don't affect the roots.
    if pfr.p > pfr.m
        pfr = PartialFracRational(pfr.λ[1:pfr.m], pfr.a, pfr.b)
    end
    @assert pfr.p ≤ pfr.m

    knownroots = ComplexF64[]
    f1 = derivative(pfr,1)
    f2 = derivative(pfr,2)

    # Functions to compute the numerator and denominator, and also r,
    # where r(x) ≡ denom'(x)/denom(x), and its derivative r1 ≡ dr/dx.
    denom(x) = prod([x-r for r in pfr.λ])
    numer(x) = pfr(x)*denom(x)
    r(x) = sum([1.0/(x-ri) for ri in pfr.λ])
    r1(x) = -sum([1.0/(x-ri)^2 for ri in pfr.λ])

    for i=1:length(rough)
        n = pfr.q-length(knownroots)

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
degree (`pfr.q` ≳ 5-10), the computation by coefficients can lead to poor conditioning
and dubious results.
"""
function roots_poly_method(pfr::PartialFracRational)
    p = Polynomial(pfr.b)
    for j=1:pfr.m
        p *= Polynomial([-pfr.λ[j],1])
    end
    for i=1:pfr.m
        p1 = Polynomial(pfr.a[i])
        for j=1:pfr.m
            i==j && continue
            p1 *= Polynomial([-pfr.λ[j],1])
        end
        p += p1
    end
    roots(real(p))
end

"""
    partial_frac_decomp(num, poles::Vector)

Compute the partial fraction decomposition of num / Π_i=1:n (z-α[i])
as ∑ i=1:n r[i]/(z-α[i]). Return the coefficient vector `r`.

Uses the general relationship P(z)/Q(z) = ∑_i=1:n P(α[i])/Q'(α[i]) * 1/(z-α[i])
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
    # The pfr is a model of F(z) = N(z)/D(z)+R(z)
    # Now find a model for F(z)/R(z) = N(z)/ [D(z)R(z)] + 1 as a partial fraction
    # Once it's a full partial fraction, it will have new poles (the roots of R) but
    # the same roots.
    if pfr.q < pfr.m
        return roots_pfrac0(pfr.a, pfr.λ[1:pfr.m])
    elseif pfr.q == pfr.m
        @assert length(pfr.b) == 1
        return roots_pfrac(pfr.a, pfr.λ[1:pfr.m], -pfr.b[end])
    end

    # q>m
    β = roots(Polynomial(real(pfr.b)))
    if pfr.m==0
        return β
    end

    # q>m>0
    α = pfr.b[end]
    a = pfr.a / α
    bc = zeros(ComplexF64, pfr.q)
    for i=1:pfr.m
        poles = [pfr.λ[i], β...]
        pfc = partial_frac_decomp(a[i], poles)
        bc[i] = pfc[1]
        bc[end-(pfr.q-pfr.m)+1:end] .+= pfc[2:end]
    end
    roots_pfrac1(-bc, vcat(pfr.λ, β))
end


"""
    roots_pfrac0(w::AbstractVector, x::AbstractVector)

Returns roots of a partial fraction that sums to zero, specifically
the `n-1` solutions `p` to the equation:

`0 = Σ i=1:n w[i]/(p-x[i])`

There are `n-1` solutions, where `n` is the length of the vectors of weights `w` and of nodes `x`.
"""
function roots_pfrac0(w::AbstractVector, x::AbstractVector)
    n = length(w)
    if n != length(x)
        throw(ErrorException("roots_pfrac0(w, x) has length(w)=$(length(w)) != length(x)=$(length(x))."))
    end
    M = diagm(x) - w/sum(w)*transpose(x)
    v = eigvals(M)
    # The vector v will contain the (n-1) desired roots, plus 0. Remove the 0.
    _, idx = findmin(abs2.(v))
    deleteat!(v, idx)
    v
end

"""
    roots_pfrac1(w::AbstractVector, x::AbstractVector)

Returns roots of a partial fraction that sums to one, specifically
the solutions `p` to the equation:

`1 = Σ i=1:n w[i]/(p-x[i])`

There are `n` solutions, where `n` is the length of the vectors of weights `w` and of nodes `x`.
"""
function roots_pfrac1(w::AbstractVector, x::AbstractVector)
    n = length(w)
    if n != length(x)
        throw(ErrorException("roots_pfrac1(w, x) has length(w)=$(length(w)) != length(x)=$(length(x))."))
    end
    M = diagm(x) + w*ones(n)'
    r = eigvals(M)
    for i=1:n
        # Take up to 3 Newton steps
        for iter = 1:3
            f = sum(w./(r[i].-x))-1
            f1 = -sum(w./(r[i].-x).^2)
            step = f/f1
            r[i] -= step
            abs(step) < 1e-13*max(1, abs(r[i])) && break
        end
    end
    r
end

"""
    roots_pfrac(w::AbstractVector, x::AbstractVector, s::Number)

Returns roots of a partial fraction that sums to `s`, specifically
the solutions `p` to the equation:

`s = Σ i=1:n w[i]/(p-x[i])`

There are `n` solutions, where `n` is the length of the vectors of weights `w` and of nodes `x`.
When `s==0`, however, we omit the implied solution at `p → ∞` and return the other `n-1` roots.
"""
function roots_pfrac(w::AbstractVector, x::AbstractVector, s::Number)
    if s==0
        return roots_pfrac0(w, x)
    end
    roots_pfrac1(w/s, x)
end
