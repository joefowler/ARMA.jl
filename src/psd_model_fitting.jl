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

include("BarycentricRational.jl")
include("weightedAAA.jl")
include("PartialFracRational.jl")
include("vector_fitting.jl")

fit_psd(ω::AbstractVector, PSD::Function, pulsemodel::AbstractVector, p, q=-1) = fit_psd(PSD.(ω), pulsemodel, p, q)

function fit_psd(PSD::AbstractVector, pulsemodel::AbstractVector, p, q=-1; invert=false)
    if p<0
        throw(DomainError("fit_psd got AR order p=$p, requires at least 0."))
    end
    if q<0
        q = p # Default to an order ARMA(p,p) model if q not given
    end

    N = length(PSD)
    ω = LinRange(0, π, N)
    ωstep = π/(N-1)
    z = cos.(ω)

    window = 1.0 .- abs.(LinRange(-1, 1, length(pulsemodel)))
    pulseFT2 = abs2.(rfft(pulsemodel.*window)/maximum(pulsemodel))
    if N != length(pulseFT2)
        throw(DimensionMismatch("fit_psd: length(PSD) $N != length(rfft(pulsemodel)) $(length(pulseFT2))"))
    end

    if invert
        PSD = 1.0 ./ PSD
        PSD[1] = 2PSD[2]-PSD[3]
        w = pulseFT2 .* PSD
    else
        PSD = copy(PSD)
        PSD[1] = 2PSD[2]-PSD[3]
        w = pulseFT2 ./ PSD.^3
    end

    # Don't let the DC bin have ZERO weight, else model likes to go negative, particularly
    # If there's lots of "action" (poles) near ω=0, or cos(ω)=1.
    w[1] = w[2]*.2

    # Signal models tend to put very little weight at ω=π (like, 1e-9x the max weight).
    # This can lead to terrible fits up there, as well as illegal roots. Prevent both by checking
    # 1. ...for minimum model value < 0.2x the minimum target function value.
    # 2. ...for illegal roots
    # If either are found, boost the minimum weight and repeat.
    minpsd = minimum(PSD)
    while true
        aaa_hybrid = aaawt(z, PSD, w, p)
        vfit1 = vectorfit(z, PSD, w, aaa_hybrid.poles, q)
        vfit = make_poles_legal(vfit1, z, PSD, w)
        # if !(vfit1 === vfit)
        #     vfit2 = vectorfit(z, PSD, w, aaa_hybrid.poles, q)
        #     vfit = make_poles_legal(vfit2, z, PSD, w)
        # end
        minmodel = minimum(real(vfit(z)))
        if minmodel > 0.2*minpsd
            cosroots = RCPRoots(roots(vfit))
            illegal_roots = illegal_RPs(cosroots)
            if length(illegal_roots) == 0
                # Pass tests 1 and 2: success.
                model = ARMAModel(vfit)
                return vfit, model
            end
        end
        w .+= 2minimum(w)  # Triple min weight each iteration
    end
end

"""
    illegal_RPs(rp)

Return a subset of `rp`: each value of `rp` that is both real and with absolute value ≤ 1.
The argument `rp` may be an `AbstractVector` or an `RCPRoots` object.
"""
illegal_RPs(rp::AbstractVector) = illegal_RPs(RCPRoots(rp))
function illegal_RPs(rp::RCPRoots)
    # Roots or poles are illegal if the absolute value is ≤1 AND if they are real
    illegal = abs.(rp) .≤ 1
    illegal[1:ncomplex(rp)] .= false
    rp[illegal]
end

"""
    make_roots_legal(vfit::PartialFracRational)

Return a new version of `vfit` in which all roots δ are "legal", i.e. either non-real, or having |δ|>1.

If `vfit`'s roots are already all legal, it is returned unchanged. If not, a constant is added to
the function to make them legal. This is not guaranteed to provide a good fit, of course, but it is legal.
The new `vfit`, of the same numerator and denominator degree, is returned.

At the moment, this can't be done if `vfit.m<vfit.n` because such rational functions don't have a
constant. Other strategies for that case are TBD.
"""
function make_roots_legal(vfit::PartialFracRational)
    while true
        cosroots = RCPRoots(roots(vfit))
        illegal_roots = illegal_RPs(cosroots)
        if length(illegal_roots) == 0
            return vfit, cosroots
        end
        # println("Have to fix $(length(illegal_roots)) illegal roots: $illegal_roots")

        if vfit.m ≥ vfit.n
            # Strategy, when m≥n: add a constant to vfit until roots are legal.
            r = illegal_roots
            midpts = 0.5*(r[1:end-1] .+ r[2:end])
            testpts = vcat(midpts, [-1,0,1])
            f = real(vfit(testpts))
            b = copy(vfit.b)
            b[1] -= 1.01*minimum(f)
            vfit = PartialFracRational(vfit.λ, vfit.a, b)
        else
            # Strategy, when m<n: I don't have one!
            @show cosroots
            @show vfit
            throw(ErrorException("No strategy for making roots legal for m<n."))
        end
    end
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
    illegal_poles = illegal_RPs(λ)
    Nbad = length(illegal_poles)
    if Nbad == 0
        return vfit
    end
    # println("Have to fix $Nbad illegal poles: $illegal_poles")

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
    for p in λ
        if p in illegal_poles
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
        WM = W*M
        WWM = W*WM
        param = WM\Wf
        model = real(M*param)
        n = length(legalλ)
        importance = (abs2.(WWM[:,1:n]*Diagonal(param[1:n])))'*ones(N)
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
        newpole = 1/cos(2ωstep)
        legalλ = RCPRoots([legalλ..., newpole])
        Mpf = [Mpf 1.0 ./(z.-newpole)]
        M = [Mpf Mrem]
    end
    @assert length(legalλ) == vfit.n

    param = (W*M)\Wf
    @assert length(param) == vfit.m+1
    a = param[1:vfit.n]
    b = param[vfit.n+1:end]
    # Make residues of non-real conjugate pair poles be conjugate pairs
    for i=1:2:ncomplex(legalλ)
        reala = 0.5*(real(a[i]+a[i+1]))
        imaga = 0.5*(imag(a[i]-a[i+1]))
        a[i] = reala+imaga*1im
        a[i+1] = reala-imaga*1im
    end
    # Make residues of real poles be real
    for i=1+ncomplex(legalλ):vfit.n
        a[i] = real(a[i])
    end
    PartialFracRational(legalλ.z, a, real(b))
end
