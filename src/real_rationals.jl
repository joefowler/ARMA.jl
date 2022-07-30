include("RCPRoots.jl")

using Polynomials


"""
    struct RealRational

Represents an order `(m,n)` rational function in polynomial-over-polynomial form. That is, as

    f(z) = (Σ i=0:m θ_i*z^i) / (Σ i=0:n b[i]*ϕ_i*z^i)

- `θ[0:m]` are the coefficients of the numerator polynomial
- `ϕ[0:n]` are the coefficients of the denominator polynomial (we rescale θ and ϕ to enforce ϕ[0]=1)
- `m,n` are the degree of the numerator and denominator polynomials, respectively

# Constructors

    RealRational(θ, ϕ)
    RealRational(zroots, zpoles, f0)

# Arguments
- `θ`: the numerator as an `AbstractPolynomial` or a vector of coefficients of one.
- `ϕ`: the denominator as an `AbstractPolynomial` or a vector of coefficients of one.
- `zroots`: the roots of `f` and `θ`.
- `zpoles`: the poles of `f` (≡ the roots of `ϕ`).
- `f0`: the value of `f(0)` is required to scale `f` when only roots and poles are given.
"""
struct RealRational{T<:AbstractPolynomial}
    θ::T
    ϕ::T
    m::Int
    n::Int
    zroots::RCPRoots
    zpoles::RCPRoots

    # function RealRational{T}(θ::AbstractPolynomial{T}, ϕ::AbstractPolynomial{T},
    #     zroots::RCPRoots,zpoles::RCPRoots) where T <: Real
    function RealRational(θ::AbstractPolynomial, ϕ::AbstractPolynomial,
        zroots::RCPRoots, zpoles::RCPRoots)
        m = degree(θ)
        n = degree(ϕ)
        @assert m ≥ 0
        @assert n ≥ 0
        @assert length(zroots) == m
        @assert length(zpoles) == n
        θ, ϕ = promote(θ, ϕ)
        T = typeof(θ)
        new{T}(θ,ϕ,m,n,zroots,zpoles)
    end
end

# Construct from θ and ϕ polynomial representation, either as polynomials or coefficients thereof
RealRational(θ::AbstractVector, ϕ::AbstractVector) = RealRational(Polynomial(θ), Polynomial(ϕ))
RealRational(θ::AbstractPolynomial, ϕ::AbstractVector) = RealRational(θ, Polynomial(ϕ))
RealRational(θ::AbstractVector, ϕ::AbstractPolynomial) = RealRational(Polynomial(θ), ϕ)
function RealRational(θ::AbstractPolynomial, ϕ::AbstractPolynomial)
    zroots = RCPRoots(roots(θ))
    zpoles = RCPRoots(roots(ϕ))
    # Normalize the input coefficient vectors
    scale = ϕ.coeffs[0]
    θ = θ ./ scale
    ϕ = ϕ ./ scale
    RealRational(θ,ϕ,zroots,zpoles)
end

function RealRational(zroots::AbstractVector,zpoles::AbstractVector,f0::Real)
    θ = fromroots(zroots)
    ϕ = fromroots(zpoles)
    θ = θ.*(f0/θ[0])
    ϕ = ϕ./ϕ[0]
    RealRational(θ,ϕ,RCPRoots(zroots),RCPRoots(zpoles))
end

Base.:*(scale::Number, rr::RealRational) = Base.:*(rr, scale)
Base.:*(rr::RealRational, scale::Number) = RealRational(rr.θ*scale, rr.ϕ, rr.zroots, rr.zpoles)

(rr::RealRational)(z) = rrat_eval(z, rr)

"""
    rrat_eval(z, rr::RealRational)

Evaluate the real-coefficient rational function `rr` at `z` (scalar or array).
"""
function rrat_eval(z::Number, rr::RealRational)
    f = evalpoly(z, rr.θ) / evalpoly(z, rr.ϕ)
end
function rrat_eval(z::AbstractVector, rr::RealRational)
    f = evalpoly.(z, rr.θ) ./ evalpoly.(z, rr.ϕ)
end

include("PartialFracRational.jl")
