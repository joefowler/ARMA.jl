module ARMA

using Polynomials, NLsolve

export
    estimate_covariance,
    ARMAModel,
    generate_noise,
    model_covariance,
    model_psd,
    toeplitz_whiten

# ARMA.jl includes the basic ARMA models and their use
# model_selection.jl includes tools for choosing a model (p,q) order
#   and for fitting the best model of that order.

include("model_selection.jl")

"An Autoregressive Moving-Average model of order (p,q)."

type ARMAModel
    p         ::Int
    q         ::Int
    sigma     ::Float64
    roots_    ::Vector{Complex128}
    poles     ::Vector{Complex128}
    thetacoef ::Vector{Float64}
    phicoef   ::Vector{Float64}
    covarIV   ::Vector{Float64}
    expbases  ::Vector{Complex128}
    expampls  ::Vector{Complex128}

    function ARMAModel(p,q,sigma,roots_,poles,thetacoef,phicoef,covarIV,expbases,expampls)
        @assert p == length(poles)
        @assert q == length(roots_)
        @assert all(abs2(poles) .> 1)
        @assert all(abs2(roots_) .> 1)
        @assert sigma >= 0.0
        @assert p+1 == length(phicoef)
        @assert q+1 == length(thetacoef)
        @assert p == length(expbases)
        @assert p == length(expampls)
        @assert 1+max(q, p) == length(covarIV)
        new(p,q,sigma,roots_,poles,thetacoef,phicoef,covarIV,expbases,expampls)
    end
end

"Go from theta,phi polynomials to the sum-of-exponentials representation.

Returns (covar_initial_values, exponential_bases, exponential_amplitudes).
"
function _covar_repr(thetacoef::Vector, phicoef::Vector)
    roots_ = roots(Poly(thetacoef))
    poles = roots(Poly(phicoef))
    expbases = 1.0 ./ poles
    q = length(roots_)
    p = length(poles)
    n = max(p,q)

    # Find the initial, exceptional values
    phi = zeros(Float64, n+1)
    phi[1:p+1] = phicoef
    P = zeros(Float64, n+1, n+1)
    for r=1:n+1
        for c=1:r
            P[r,c] = phi[r-c+1]
        end
    end
    theta = zeros(Float64, n+1)
    theta[1:q+1] = thetacoef
    psi = P \ theta

    T = zeros(Float64, n+1, n+1)
    for r=1:n+1
        for c=1:n+2-r
            T[r,c] = theta[c+r-1]
        end
    end
    P2 = zeros(Float64, n+1, n+1)
    for r=1:n+1
        for i=1:p+1
            c = r-i+1
            if c<1; c = 2-c; end
            P2[r,c] += phi[i]
        end
    end

    gamma = P2 \ (T*psi)
    XI = Array{Complex128}(p, p)
    lowestpower = p >= q ? 1 : 1+q-p
    for c=1:p
        XI[1,c] = expbases[c] ^ lowestpower
        for r=2:p
            XI[r,c] = expbases[c] * XI[r-1,c]
        end
    end
    expampls = XI \ model_covariance(gamma, phicoef, p+lowestpower)[end+1-p:end]
    gamma, expbases, expampls
end

"Constructors for the ARMAModel object"
# Construct from roots-and-poles representation
function ARMAModel(sigma::Float64, roots_::Vector, poles::Vector)
    q = length(roots_)
    p = length(poles)
    thetac = poly(roots_).a
    phic = poly(poles).a
    thetacoef = sigma * thetac / thetac[1]
    phicoef = phic / phic[1]

    covarIV, expbases, expampls = _covar_repr(thetacoef, phicoef)
    ARMAModel(p,q,sigma,roots_,poles,thetacoef,phicoef,covarIV,expbases,expampls)
end


# Construct from theta and phi polynomial representation
function ARMAModel(thetacoef::Vector, phicoef::Vector)
    sigma = thetacoef[1] / phicoef[1]
    roots_ = roots(Poly(thetacoef))
    poles = roots(Poly(phicoef))
    q = length(roots_)
    p = length(poles)

    covarIV, expbases, expampls = _covar_repr(thetacoef, phicoef)
    ARMAModel(p,q,sigma,roots_,poles,thetacoef,phicoef,covarIV,expbases,expampls)
end

# Construct ARMAModel from a sum-of-exponentials representation
function ARMAModel(bases::Vector, amplitudes::Vector, covarIV::Vector, q::Int)
    p = length(bases)
    @assert p == length(amplitudes)
    @assert length(covarIV) >= q-p+1

    # Find the covariance from lags 0 to p+q. Call it gamma
    gamma = zeros(Float64, 1+p+q)
    t = collect(0:p+q)
    for i=1:p
        gamma += real(amplitudes[i] * (bases[i] .^ t))
    end
    gamma[1:length(covarIV)] = covarIV

    # Find the phi polynomial
    poles = 1.0 ./ bases
    phic = poly(poles).a
    phicoef = phic / phic[1]

    # Find the nonlinear system of equations for the theta coefficients
    LHS = zeros(Float64, 1+q)
    for t=0:q
        for i=1:p+1
            for j=1:p+1
                LHS[t+1] += phicoef[i]*phicoef[j]*gamma[1+abs(t+i-j)]
            end
        end
    end
    @show gamma
    @show LHS

    # Solve the system! TODO
    function f!(x, residual)
        q = length(residual)-1
        for t = 0:q
            residual[t+1] = LHS[t+1]
            for j = 0:q-t
                residual[t+1] -= x[j+1]*x[j+t+1]
            end
        end
        residual
    end
    thetacoef = ones(Float64, q+1)
    #df = DifferentiableMultivariateFunction(f!)
    results = nlsolve(f!, thetacoef)
    roots_ = roots(Poly(results.zero))

    # Now a clever trick: any roots of the MA polynomial that are INSIDE
    # the unit circle can be replaced by 1/r and yield the same covariance.
    @show abs2(roots_)
    for i=1:q
        if abs2(roots_[i]) < 1
            roots_[i] = 1.0/roots_[i]
        end
    end
    # Of course, the polynomial needs to be re-made in case any roots moved
    thetac = poly(roots_).a
    sigma = sqrt(real(gamma[1]))
    thetacoef = sigma * thetac / thetac[1]
    @show abs2(roots_)


    ARMAModel(p,q,sigma,roots_,poles,thetacoef,phicoef,gamma[1:1+max(p,q)],bases,amplitudes)
end

"generate a simulated noise timeseries from an ARMAModel of length N"
function generate_noise(m::ARMAModel, N::Int)
    # eps = white N(0,1) noise; x = after MA process; z = after inverting AR
    eps = randn(N+m.q)
    eps[1:m.p] = 0
    x = zeros(Float64, N)
    z = zeros(Float64, N)
    for i=1:m.q+1
        x += eps[i:end+i-m.p-1] * m.thetacoef[i]
    end
    for j=1:m.p
        z[j] = x[j]
        for i = 2:j
            z[j] -= m.phicoef[i] * z[j-i+1]
        end
    end
    for j=1+m.p:N
        z[j] = x[j]
        for i = 2:m.p+1
            z[j] -= m.phicoef[i] * z[j-i+1]
        end
    end
    z
end

"The ARMA model's model covariance function, from lags 0 to N-1"
function model_covariance(covarIV::Vector, phicoef::Vector, N::Int)
    covar = zeros(Float64, N)
    covar[1:length(covarIV)] = covarIV
    @assert phicoef[1] == 1.0
    for i = length(covarIV)+1:N
        for j = 1:length(phicoef)-1
            covar[i] -= phicoef[j+1] * covar[i-j]
        end
    end
    covar
end
model_covariance(m::ARMAModel, N::Int) = model_covariance(m.covarIV, m.phicoef, N)


"The ARMA model's power spectral density function"
function model_psd(m::ARMAModel, N::Int)
    freq = collect(linspace(0,0.5,N))
    model_psd(m, freq)
end

function model_psd(m::ARMAModel, freq::Vector)
    z = exp(-2im*pi *freq)
    numer = m.thetacoef[1]
    for i=1:m.q
        numer += m.thetacoef[i+1] * (z.^i)
    end
    denom = m.phicoef[1]
    for i=1:m.p
        denom += m.phicoef[i+1] * (z.^i)
    end
    abs2(numer ./ denom)
end

"Approximately whiten the timestream using a Toeplitz matrix (so
that a zero-padded delay of the input timestream is equivalent to a
zero-padded delay of the output).

No Toeplitz matrix has the ability to make the input exactly white,
but for many purposes, the time-shift property is more valuable than
that exact whitening."
function toeplitz_whiten(m::ARMAModel, timestream::Vector)
    N = length(timestream)
    white = zeros(Float64, N)

    # First, multiply the input by the AR matrix (a banded Toeplitz
    # matrix with the phi coefficients on the diagonal and first p
    # subdiagonals).
    # The result is the MA matrix times the whitened data.
    MAonly = m.phicoef[1] * timestream
    for i=1:m.p
        MAonly[1+i:end] .+= m.phicoef[i+1] * timestream[1:end-i]
    end

    # Second, solve the MA matrix (also a banded Toeplitz matrix with
    # q non-zero subdiagonals.)
    white[1] = MAonly[1] / m.thetacoef[1]
    if N==1
        return white
    end
    for i = 2:min(m.q, N)
        white[i] = MAonly[i]
        for j = 1:i-1
            white[i] -= white[j]*m.thetacoef[1+i-j]
        end
        white[i] /= m.thetacoef[1]
    end
    for i = m.q+1:N
        white[i] = MAonly[i]
        for j = i-m.q:i-1
            white[i] -= white[j]*m.thetacoef[1+i-j]
        end
        white[i] /= m.thetacoef[1]
    end
    white
end

function toeplitz_whiten!(m::ARMAModel, timestream::Vector)
    timestream[1:end] = toeplitz_whiten(m, timestream)
end

end # module
