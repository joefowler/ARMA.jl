# Allow user to fit for an ARMA model in power spectrum, not autocorrelation domain. An ARMA(p,q)
# noise model yields a noise power spectrum that is rational of order(q-over-p) in cos(ω) and is
# nonzero and finite for all ω∈[0,π] or cos(ω)∈[-1,1].
#
# Proceeds in several steps to find f(x=cos(ω)) ≈ PSD(ω), broadly:
# 1. Find the poles of f
# 1a. Find poles of an AAA mixed interpolation/fit.
# 1b. Find poles of best-fit rational function by iteration, with previous (AAA) step as initial guess.
# 1c. "Fix" poles by moving off the real line in [-1,1] to make f(x) be nonzero and finite at all real cos(ω)∈[-1,1].
# 2. Find the roots of f

include("RCPRoots.jl")
include("BarycentricRational.jl")
include("weightedAAA.jl")
include("PartialFracRational.jl")
include("vector_fitting.jl")

fit_psd(ω::AbstractVector, PSD::Function, pulsemodel::AbstractVector, p, q=-1) = fit_psd(PSD.(ω), pulsemodel, p, q)

function fit_psd(PSD::AbstractVector, pulsemodel::AbstractVector, p, q=-1)
    if p<0
        throw(DomainError("fit_psd got AR order p=$p, requires at least 0."))
    end
    if q<0
        q = p # Default to an order ARMA(p,p) model if q not given
    end

    N = length(PSD)
    ω = LinRange(0, π, N)
    ωstep = ω[2]
    z = cos.(ω)
    pulseFT2 = abs2.(rfft(pulsemodel)/maximum(pulsemodel))
    if N != length(pulseFT2)
        throw(DimensionMismatch("fit_psd: length(PSD) $N != length(rfft(pulsemodel)) $(length(pulseFT2))"))
    end

    clf()
    w = pulseFT2 ./ PSD.^3
    # Don't let the DC bin have ZERO weight, else model likes to go negative, particularly
    # If there's lots of "action" (poles) near ω=0, or cos(ω)=1.
    w[1] = w[2]*.01

    aaa_hybrid = aaawt(z, PSD, w, p)
    vfit1 = vectorfit(z, PSD, w, aaa_hybrid.poles, q)
    loglog(ω, PSD, ".-", label="Input data")
    loglog(ω, vfit1(z), "--", label="Vfit 1 (illegal poles)")

    vfit = make_poles_legal(vfit1, z, PSD, w)
    loglog(ω, vfit(z), label="Vfit 2 (models legal)")
    @show vfit.λ
    ma_roots = find_roots(vfit)
    @show ma_roots
    # ma_roots = make_roots_legal(ma_roots)
    # @show ma_roots

    zpoles = exp.(acosh.(vfit.λ))
    zroots = exp.(acosh.(complex(ma_roots[1:end])))
    println()
    @show zpoles
    @show zroots
    legend()
    var = mean(PSD)
    model = ARMAModel(vfit)
    vfit, model
end

make_poles_legal(vfit::PartialFracRational, z::AbstractVector, PSD::AbstractVector; angletol=1e-13) =
    make_poles_legal(vfit, z, PSD, ones(Float64, length(z)); angletol)

"""
    make_poles_legal(vfit::PartialFracRational, z, PSD, wt)

Return a new version of `vfit` in which all poles λ are "legal", i.e. either non-real, or having |λ|>1.

If `vfit`'s poles are already all legal, it is returned unchanged. If not, the illegal ones are duplicated
and moved to conjugate-pair locations a reasonable, small increment off the real line. Then, (weighted) linear
fits are made and poles removed one real pole or a pair of conjugate poles according to those with the lowest
total (weighted) power in the model, until the right number of poles remain. If this procedure yields one
pole too few (because a pair was removed), then a "bonus pole" is added to the set. The final best fit
with the same number of poles as `vfit` is converted to a `PartialFracRational` and returned.

The fit is done with the power spectrum `PSD` sampled at `z=cos.(ω)` and with statistical weights `wt`.
"""
function make_poles_legal(vfit::PartialFracRational, z::AbstractVector, PSD::AbstractVector, wt::AbstractVector; angletol=1e-13)
    λ = RCPRoots(vfit.λ)
    # Poles are illegal if the absolute value is ≤1 AND if they are real
    pole_is_illegal = abs.(λ) .≤ 1
    pole_is_illegal[1:ncomplex(λ)] .= false
    Nbad = sum(pole_is_illegal)
    if Nbad == 0
        return vfit
    end

    N = length(z)
    ωstep = π/N
    if N != length(PSD)
        throw(DimensionMismatch("fit_psd: length(z) $N != length(PSD) $(length(PSD))"))
    elseif N != length(wt)
        throw(DimensionMismatch("fit_psd: length(z) $N != length(wt) $(length(wt))"))
    end

    # Build a vector of _legal_ poles, including all poles already legal, plus the pairs p±ϵ*1im
    # for a sensible, small ϵ that yields a Lorentzian of width ωstep at the given frequency.
    # This vector will be longer than λ by the number of illegal poles in λ.
    legalλ = ComplexF64[]
    for (p, illegal) in zip(λ, pole_is_illegal)
        if illegal
            bonus = 1e-7  # Ensure nonzero step even if abs(p) == 1.
            p_tweaked = complex.(p) + ωstep*sqrt(1+bonus-abs2(p))*1im
            push!(legalλ, p_tweaked)
            push!(legalλ, conj(p_tweaked))
        else
            push!(legalλ, p)
        end
    end
    legalλ = RCPRoots(legalλ)

    Mpf = hcat([1.0 ./ (z.-L) for L in legalλ]...)
    zscaled = 2(z.-vfit.polyMin)/(vfit.polyMax-vfit.polyMin) .- 1
    Mrem = ones(eltype(zscaled), N, vfit.m+1-vfit.n)
    for i=1:vfit.m-vfit.n
        Cheb = ChebyshevT([zeros(i)..., 1.0])
        Mrem[:, i+1] = Cheb.(zscaled)
    end
    M = [Mpf Mrem]
    W = Diagonal(sqrt.(wt))
    Wf = W*PSD
    while length(legalλ) > vfit.n
        param = (W*M)\Wf
        model = real(M*param)
        n = length(legalλ)
        importance = (abs2.(Diagonal(wt)*M[:,1:n]*Diagonal(param[1:n])))'*ones(N)
        least = findfirst(importance .== minimum(importance))
        least === nothing && @show importance, minimum(importance)
        if least ≤ ncomplex(legalλ)  # delete a complex pair
            if least%2 == 0
                least -= 1
            end
            keep = ones(Bool, size(Mpf)[2])
            keep[least:least+1] .= false
            legalλ = RCPRoots(legalλ.z[keep])
            Mpf = Mpf[:, keep]
        else
            keep = ones(Bool, size(Mpf)[2])
            keep[least] = false
            legalλ = RCPRoots(legalλ.z[keep])
            Mpf = Mpf[:, keep]
        end
        M = [Mpf Mrem]
    end
    if length(legalλ) < vfit.n
        @assert length(legalλ) == vfit.n-1
        newpole = 1/cos(ωstep)
        legalλ = RCPRoots([legalλ..., newpole])
        Mpf = [Mpf 1.0 ./(z.-newpole)]
        M = [Mpf Mrem]
    end
    @assert length(legalλ) == vfit.n

    param = (W*M)\Wf
    @assert length(param) == vfit.m+1
    a = param[1:vfit.n]
    b = param[vfit.n+1:end]
    for i=1:2:ncomplex(legalλ)
        reala = 0.5*(real(a[i]+a[i+1]))
        imaga = 0.5*abs(imag(a[i]-a[i+1]))
        a[i] = reala+imaga*1im
        a[i+1] = reala-imaga*1im
    end
    PartialFracRational(legalλ.z, a, real(b))
end


"""
    partial_frac(num, poles::Vector)

Compute the partial fraction decomposition of num / Π_i=1:n (z-α[i])
as ∑ i=1:n r[i]/(z-α[i]). Return the coefficient vector `r`.

Generally P(z)/Q(z) = ∑ P(α[i])/Q'(α[i]) * 1/(z-α[i])
"""
function partial_frac(num::Number, poles::AbstractVector)
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

function find_roots(vfit::PartialFracRational; steptol=1e-14)
    # The vfit is a model of F(z)=N(z)/D(z)+R(z)
    # Now find a model for F(z)/R(z) = N(z)/ [D(z)R(z)] + 1 as a partial fraction
    # Once it's a full partial fraction, it will have new poles (the roots of R) but
    # the same roots.
    p, q = vfit.n, vfit.m
    rough_roots = nothing
    if q<p
        rough_roots = roots_pfrac0(vfit.a, vfit.λ)
    elseif q==p
        rough_roots = roots_pfrac(vfit.a, vfit.λ, -vfit.b[end])
    else
        α = vfit.b[end]
        a = vfit.a / α
        bc = zeros(ComplexF64, q)
        β = ComplexF64[]
        β = chebyshev_roots(real(vfit.b))
        if p==0
            rough_roots = β
        else
            for i=1:p
                poles = [vfit.λ[i], β...]
                pfc = partial_frac(a[i], poles)
                bc[i] = pfc[1]
                bc[end-(q-p)+1:end] .+= pfc[2:end]
            end
            rough_roots = roots_pfrac1(-bc, vcat(vfit.λ, β))
        end
    end

    # Poor conditioning can make the eigenvalue method yield imperfect values. "Polish" those results
    # by a few Newton-Raphson steps
    remainder = ChebyshevT(vfit.b)
    dremainder = derivative(remainder)
    F = vfit
    function dFdx(x)
        f0 = 0.0im
        for (ρ, λ) in zip(vfit.a, vfit.λ)
            f0 -= ρ/(x-λ)^2
        end
        f0 + evalpoly.(complex(x), dremainder, false)
    end

    r = copy(rough_roots)
    for i=1:length(r)
        # Try Newton steps, up to 12.
        x = r[i]
        prevF = abs(F(x))
        if abs(F(real(x))) < prevF
            x = real(x)
            prevF = abs(F(x))
        end
        for iter=1:12
            # Careful: very small _intended_ steps can actually yield no change at all
            # to floating-point precision. Check the actual step, not the intended one.
            intended_step = -F(x)/dFdx(x)
            xnew = x+intended_step
            newF = abs(F(xnew))
            if newF < prevF
                actual_step = xnew-x
                min_step = 3eps(abs(x))
                x, prevF = xnew, newF
                abs(actual_step) < steptol*abs(x) || abs(actual_step) < min_step && break
            else
                break
            end
        end
        if eltype(r) <: Real
            r[i] = real(x)
        else
            r[i] = x
        end
    end
    RCPRoots(r)
end
