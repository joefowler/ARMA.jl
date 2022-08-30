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
        # Move the last (real) pole from the end of the list into the end of the residues' portion.
        λ = λ.z
        λ[n_residues:end] = circshift(λ[n_residues:end],1)
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
function PairedPartialFracRational(pfr::PartialFracRational)
    unum = Float64[]
    udenom = Float64[]

    for i=1:2:pfr.m-1
        # For now, assert that poles/residues are sorted into complex pairs first, or are real.
        @assert isapprox(sin(angle(pfr.λ[i]+pfr.λ[i+1])), 0; atol=1e-12)
        @assert isapprox(sin(angle(pfr.a[i]+pfr.a[i+1])), 0; atol=1e-12)
        push!(unum, real(sum(pfr.a[i:i+1])))
        push!(unum, -real(pfr.a[i]*pfr.λ[i+1]+pfr.a[i+1]*pfr.λ[i]))
        push!(udenom, real(sum(pfr.λ[i:i+1])))
        push!(udenom, real(prod(pfr.λ[i:i+1])))
    end
    if pfr.m%2 == 1
        push!(unum, real(pfr.a[pfr.m]))
        push!(udenom, real(pfr.λ[pfr.m]))
    end

    extrapoles = RCPRoots(pfr.λ[1+pfr.m:end])
    extrau = Float64[]
    for i=1:2:length(extrapoles)-1
        u1 = real(sum(extrapoles[i:i+1]))
        u0 = real(prod(extrapoles[i:i+1]))
        append!(extrau, [u1,u0])
    end
    if length(extrapoles)%2 == 1
        append!(extrau, real(extrapoles[end]))
    end
    @assert length(unum) == pfr.m
    @assert length(pfr.b) == pfr.q-pfr.m+1
    @assert length(udenom) == pfr.m
    @assert length(extrau) == pfr.p-pfr.m
    u = vcat(unum, real(pfr.b), udenom, extrau)
    PairedPartialFracRational(u, pfr.q, pfr.p, pfr.m)
end

function PartialFracRational(ppfr::PairedPartialFracRational)
    unum = ppfr.u[1:ppfr.m]
    remainder = ppfr.u[1+ppfr.m:1+ppfr.q]
    udenom = ppfr.u[2+ppfr.q:end]
    @assert length(udenom) == ppfr.p

    a = Complex{Float64}[]
    λ = Complex{Float64}[]

    for i=1:2:ppfr.m-1
        pmterm = .5*sqrt(udenom[i]^2-4Complex(udenom[i+1]))
        push!(λ, udenom[i]*.5+pmterm)
        push!(λ, udenom[i]*.5-pmterm)
        avalues = [1 1; -λ[i+1] -λ[i]] \ unum[i:i+1]
        append!(a, avalues)
    end
    if ppfr.m%2 == 1
        push!(λ, udenom[ppfr.m])
        push!(a, unum[ppfr.m])
    end
    for i=1+ppfr.m:2:ppfr.p-1
        pmterm = .5*sqrt(udenom[i]^2-4Complex(udenom[i+1]))
        push!(λ, udenom[i]*.5+pmterm)
        push!(λ, udenom[i]*.5-pmterm)
    end
    if (ppfr.p-ppfr.m)%2 == 1
        push!(λ, udenom[end])
    end

    @assert length(a) == ppfr.m
    @assert length(λ) == ppfr.p
    PartialFracRational(λ, a, remainder)
end

#  RR <--> PPFR follow as the obvious 2-step conversions with PFR as the intermediate.
PairedPartialFracRational(rr::RealRational) = PairedPartialFracRational(PartialFracRational(rr))
RealRational(ppfr::PairedPartialFracRational) = RealRational(PartialFracRational(ppfr))
