include("RCPRoots.jl")

using Polynomials
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
        zroots::RCPRoots,zpoles::RCPRoots)
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
    scale = ϕ.coeffs[1]
    θ = θ ./ scale
    ϕ = ϕ ./ scale
    RealRational(θ,ϕ,zroots,zpoles)
end

function RealRational(zroots::AbstractVector,zpoles::AbstractVector,f0::Real)
    θ = fromroots(zroots)
    ϕ = fromroots(zpoles)
    θ = θ.*(f0/θ[1])
    ϕ = ϕ./ϕ[1]
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
    f = evalpoly(z, rr.ϕ) ./ evalpoly(z, rr.θ)
end
function rrat_eval(z::AbstractVector, rr::RealRational)
    f = evalpoly.(z, rr.ϕ) ./ evalpoly.(z, rr.θ)
end

function PartialFracRational(rr::RealRational)
    λ = rr.zpoles

    # How many partial fraction terms to compute
    npf = min(rr.n, rr.m+1)
    if (rr.n%2 == 0) && (rr.m^2 == 0) && rr.m < rr.n
        npf -= 1
    end

    @assert npf == rr.n # Not implemented yet to have fractional factors * partial fraction

    dϕdz = derivative(rr.ϕ)
    residues = 0λ
    for (j,p) in enumerate(λ)
        residues[j] = -rr.θ(p)/(p*dϕdz(p))
    end

    @assert rr.m == rr.n # Not implemented yet to have non-constant remainder
    remainder = [rr.θ.coeffs[end]/rr.ϕ.coeffs[end]]
    @show λ, residues, remainder
    PartialFracRational(λ, residues, remainder)
end
