module ARMA

# package code goes here
export
    estimate_covariance
    estimate_covariance,
    ARMAModel,

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

end # module
