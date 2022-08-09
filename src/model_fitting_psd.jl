#
# Functions to find the appropriate ARMA model from power spectrum data.
#

using OffsetArrays

"""
    innovations_estimate(autocorr::AbstractVector, p::Integer, q=-1)

Make a preliminary fit for ARMA model of the specified order from the autocorrelation.
Based on Brockwell & Davis §8.3 and §8.4. This estimate is useful as a starting point for further
refinements, such as a weighted fit to the power spectrum.

Operates in two steps. First it estimates the equivalent MA(∞) model as a MA(`n-1`) model
with coefficients ψ (`n` is the length of the input `autocorr`). Then it converts the ψ to
the equivalent mixed ARMA model, with coefficients θ and ϕ for the MA and AR parts.

# Arguments
- `autocorr`: The ARMA process autocorrelation function, as estimated from data.
- `p`: The AR order of the ARMA process.
- `q`: The MA order of the ARMA process (if `q` is not given, then let `q=p`).

# Returns

    (psd,θ,ϕ)
- `psd` is a scalar function returning the power spectrum at the given angular frequency
- `θ` is the MA polynomial
- `ϕ` is the AR polynomial

"""
function innovations_estimate(autocorr::AbstractVector, p::Integer, q=-1)
    if q < 0
        q = p
    end
    @assert p ≥ 0
    @assert q ≥ 0

    n = length(autocorr)
    v = OffsetArray(zeros(Float64, n), -1) # indices 0:n-1
    θ = zeros(Float64, n-1, n)
    v[0] = autocorr[begin]

    # Fit MA models of order m=1,2,...n-1 (n-1 is the largest we can do with n data values)
    # The order-m coefficients appear in the mth row as θ[m,1:m] (there's an implied θ[m,0]=1),
    # and the white-noise variance for each model is v[m].
    for m=1:n-1
        θ[m,m] = autocorr[m+1]/v[0]
        for k=1:m-1
            s = 0.0
            for j=0:k-1
                s += θ[m,m-j]*θ[k,k-j]*v[j]
            end
            θ[m,m-k] = (autocorr[m+1-k]-s)/v[k]
        end
        v[m] = autocorr[begin]
        for j=0:m-1
            v[m] -= θ[m,m-j]^2*v[j]
        end
    end

    # Now treat the last estimate as the leading part of an MA(∞) model with coefficients ψ.
    # Convert that model to the best mixed model ARMA(p,q). Again implied ψ[0]=1.
    ψ = θ[end,:]
    if p > 0
        Mrow1 = zeros(Float64, p)
        if q≥p
            Mrow1[:] = ψ[q:-1:q+1-p]
        else
            Mrow1[1:q] = ψ[q:-1:1]
            Mrow1[q+1] = 1.0  # the implied ψ[0]=1.
        end
        M = Toeplitz(ψ[q:q+p-1], Mrow1)
        ϕ = M\ψ[q+1:q+p]  # Eq 8.4.4
        pϕ = Polynomial(vcat(1, -ϕ))  # Our convention uses opposite sign from Brockwell & Davis.
    else
        pϕ = Polynomial([1])
    end

    theta = ψ[1:q]  # This plus following loop perform Eq 8.4.5
    for j=1:q
        for i=1:min(j-1,p)
            theta[j] -= ϕ[i]*ψ[j-i]
        end
        if j≤p
            theta[j] -= ϕ[j]
        end
    end
    pθ = sqrt(v[end])*Polynomial(vcat(1, theta))

    function psd(ω::Number)
        z = exp(1im*ω)
        zbar = conj(z)
        abs(pθ(z)*pθ(zbar)/(pϕ(z)*pϕ(zbar)))/π
    end
    psd, pθ, pϕ
end
