using ToeplitzMatrices

"""
    PartialFracRational(rr::RealRational)

Convert `rr` to a `PartialFracRational` form.
"""
function PartialFracRational(rr::RealRational)
    # How many partial fraction terms to compute?
    n_residues = min(rr.n, rr.m+1)

    # If `rr` is an even-over-even rational with fewer roots than poles, make sure the poles are
    # split into an even number with residues and an even number of extra factors (not odd/odd).
    if (rr.n%2 == 0) && (rr.m%2 == 0) && rr.m < rr.n
        n_residues -= 1
    end

    λ = rr.zpoles
    # Re-sort λ so no complex pairs are broken when we split into the [1,n_residues] + remainder.
    # Only would happen if n_residues is odd, AND there are more than n_residues complex poles.
    if (n_residues%2 != 0) && λ.ncomplex > n_residues
        @assert length(λ)%2 == 1
        # that works b/c n_residues + n_remainder == length(λ), and the 2 terms can't both be odd.
        # So push the last (real) pole into the front of the list.
        λ = circshift(λ,1)
    end
    # TODO: are there ways to prioritize which ones get a residue and a partial-fraction, vs
    # which are simply "extra factors"? For now, we'll assume it makes no difference.

    residues = zeros(Complex{Float64}, n_residues)
    dϕdz = derivative(rr.ϕ)
    firstλ = λ[1:n_residues]
    residues = rr.θ.(firstλ) ./ dϕdz.(firstλ)
    for k=1+n_residues:rr.n
        residues .*= (firstλ .- λ[k])
    end

    if n_residues > rr.m
        @assert n_residues == rr.m+1
        noremainder = Float64[]
        return PartialFracRational(λ, residues, noremainder)
    elseif n_residues == rr.m
        r0 = rr.θ[end]/rr.ϕ[end]
        remainder = [r0]
        return PartialFracRational(λ, residues, remainder)
    end

    # Here, n_residues=rr.n < rr.m, so remainder has degree rr.m-rr.n ≥ 1.
    # Do polynomial synthetic division to compute the coefficients.
    @assert n_residues == rr.n
    n_remainder = rr.m-rr.n+1
    v = zeros(eltype(rr.ϕ), n_remainder)
    if length(rr.ϕ) > n_remainder
        v[:] = rr.ϕ[end:-1:begin][1:n_remainder]
    else
        v[1:length(rr.ϕ)] = rr.ϕ[end:-1:begin]
    end
    T = TriangularToeplitz(v, :U)
    remainder = T \ rr.θ[rr.n:rr.m]
    PartialFracRational(λ, residues, remainder)
end

function RealRational(pfr::PartialFracRational)
    f0 = real(pfr(0))
    zroots = roots(pfr)
    RealRational(zroots, pfr.λ, f0)
end

# also need conversions between PFR and PPFR
# then RR <--> PPFR follow as the obvious 2-step conversions with PFR as the intermediate.
