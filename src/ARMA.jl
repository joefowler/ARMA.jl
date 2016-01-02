module ARMA

using Polynomials, NLsolve

export
    estimate_covariance,
    ARMAModel,
    generate_noise,
    model_covariance,
    model_psd

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

end # module
