using ARMA, Polynomials
using Base.Test

# 1) Test padded_length: rounds up to convenient size for FFT

@test ARMA.padded_length(1000) == 1024
@test ARMA.padded_length(16) == 16
@test ARMA.padded_length(12) == 16
@test ARMA.padded_length(11) == 12
@test ARMA.padded_length(10) == 12
@test ARMA.padded_length(9) == 10
@test ARMA.padded_length(8) == 8
for i=1:10
    j = rand(1:1025)
    @test ARMA.padded_length(j) >= j
end

# 2) Test estimate_covariance

@test estimate_covariance([0,2,0,-2]) == [2.0, 0.0, -2.0, 0.0]
u = randn(2+2^13)
r = u[3:end] + u[1:end-2] + 2*u[2:end-1]
cv = estimate_covariance(r, 20)
@test abs(cv[1] - 6) < 1
@test abs(cv[2] - 4) < 1
@test abs(cv[3] - 1) < 1
for lag = 4:19
    @test abs(cv[lag]) < 1
end

@test estimate_covariance(u, 10) == estimate_covariance(u, 10, div(length(u), div(length(u)-1+150,150)))
@test length(estimate_covariance(u)) == length(u)

# 3) Basic tests of ARMAModel constructors

function similar_list(a::Vector, b::Vector, eps)
    @assert length(a) == length(b)
    for a1 in a
        if all(abs2(b - a1) .> eps^2)
            return false
        end
    end
    for b1 in b
        if all(abs2(a - b1) .> eps^2)
            return false
        end
    end
    true
end

p,q = 3,3
rs = 1+(randn(q) .^ 2)
ps = 1+(randn(p) .^ 2)
variance = 1.0
m = ARMAModel(rs, ps, variance)
@test m.p == p
@test m.q == q
n = ARMAModel(m.thetacoef, m.phicoef)
@test m.thetacoef == n.thetacoef
@test m.phicoef == n.phicoef
@test similar_list(m.roots_, n.roots_, 1e-7)
@test similar_list(m.poles, n.poles, 1e-7)

function toeplitz(c::Vector)
    N = length(c)
    t = Array{eltype(c)}(N,N)
    for i=1:N
        for j=1:i
            t[i,j] = t[j,i] = c[1+i-j]
        end
    end
    t
end

function toeplitz(c::Vector, r::Vector)
    M,N = length(c), length(r)
    t = Array{eltype(c)}(M,N)
    for i=1:M
        for j=1:i
            t[i,j] = c[1+i-j]
        end
        for j=i+1:N
            t[i,j] = r[1+j-i]
        end
    end
    t
end

# 4) Now complete tests of several models that have been worked out carefully
# on paper, as well as several that are randomly created.

# Generate 6 models of fixed parameters and order (2,0), (0,2), (1,1), (1,2), (2,1), (2,2)
thetas=Dict('A'=>[2], 'B'=>[2,2.6,.8], 'C'=>[2,1.6], 'D'=>[2,2.6,.8], 'E'=>[2,1.6], 'F'=>[2,2.6,.8])
phis = Dict('A'=>[1,-.3,-.4], 'B'=>[1], 'C'=>[1,-.8], 'D'=>[1,-.8], 'E'=>[1,-.3,-.4], 'F'=>[1,-.3,-.4])
const EPSILON = 2e-4

# And generate 6 models of random order and random parameters
for model in "GHIJKL"
    # Order will be 0<=p,q <=6.
    # Use rand^(-.3) for roots/poles. Negative power ensures abs(r)>1, and
    # the 0.3 power concentrates the values near the unit circle.
    p = rand(0:6)
    q = rand(0:6)
    if p+q==0; p=q=5; end  # Don't test ARMA(0,0) model!
    roots_ = rand(q) .^ (-.3)
    poles = rand(p) .^ (-.3)

    # Want one negative pole, if p>=3
    if p>2
        poles[end] *= -1
    end

    # Half the time, on larger-order models, make one pair roots and/or poles complex.
    if p>2 && rand(0:1) == 1
        poles = complex(poles)
        poles[1] = complex(real(poles[1]),real(poles[2]))
        poles[2] = conj(poles[1])
    end

    if q>2 && rand(0:1) == 1
        roots_ = complex(roots_)
        roots_[1] = complex(real(roots_[1]),real(roots_[2]))
        roots_[2] = conj(roots_[1])
    end

    # Scale theta by 0.7 to avoid lucky cancellations in the tests.
    thetas[model] = ARMA.polynomial_from_roots(roots_) * 0.7
    phis[model] = ARMA.polynomial_from_roots(poles)
    phis[model] *= 1.0/phis[model][1]
end

# Loop over all the models specified by their rational function representation
# in thetas[] and phis[]. For each model, construct it all 3 ways (theta,phi;
# roots, poles, and variance; or sum-of-exponentials). Verify that the resulting
# model has the same covariance and other key properties.

for model in "ABCDEFGHIJKL"
    thcoef = thetas[model]
    phcoef = phis[model]
    if phcoef[1] != 1.0
        thcoef /= phcoef[1]
        phcoef /= phcoef[1]
    end
    @assert phcoef[1] == 1.0
    const p = length(phcoef)-1
    const q = length(thcoef)-1
    # println("Testing model $model of order ARMA($p,$q).")

    m1 = ARMAModel(thcoef, phcoef)

    roots_ = roots(Poly(thcoef))
    poles = roots(Poly(phcoef))
    expbases = 1.0 ./ poles

    # We'll be working with q+1 equations to find initial values
    # of psi: the Taylor expansion coefficients of theta(z)/phi(z).
    # See BD (3.3.3) for the q+1 initial equations and  (3.3.4) for the
    # homogeneous equations beyond the first q+1. Careful with the sign conventions,
    # b/c BD uses phi(z) = 1 - (phi1*z + phi2*z^2 + ...), while I prefer a + sign.
    phpad = zeros(Float64, q+1)
    if q>p
        phpad[1:p+1] = phcoef
    else
        phpad[1:q+1] = phcoef[1:q+1]
    end

    psi = copy(thcoef)
    for j=1:q
        for k=1:j
            psi[j+1] -= phpad[k+1]*psi[1+j-k]
        end
    end

    # We have to solve for the first N=max(p,q)+1 values of covariance at once.
    # For these, see BD equation (3.3.8) for the first q+1 equations and (3.3.9)
    # for the remaining (p-q), if any.
    N = 1+max(p,q)
    phN = copy(phcoef)
    if q>p
        append!(phN, zeros(Float64, q-p))
    end
    A = zeros(Float64, N, N)
    for i=1:N
        for j=1:N
            col = 1+abs(j-i)
            A[i,col] += phN[j]
        end
    end
    rhs3_3_8 = zeros(Float64, N)
    for k=1:q+1 # here j,k are both 1 larger than in BD 3.3.8.
        for j=k:q+1
            rhs3_3_8[k] += thcoef[j]*psi[1+j-k]
        end
    end
    gamma = A \ rhs3_3_8

    m2 = ARMAModel(roots_, poles, gamma[1])

    if q<p
        covarIV=Float64[]
    else
        covarIV = gamma[1:1+q-p]
    end
    B = Array{Complex128}(p,p)
    for r=1:p
        for c=1:p
            B[r,c] = expbases[c]^(N-p+r-1)
        end
    end
    expampls = B \ gamma[N-p+1:N]

    m3 = ARMAModel(expbases, expampls, covarIV)

    # Check that model orders are equivalent
    # Take care with model m3, b/c it never sets q<p-1 when constructing.
    @test p == m1.p
    @test p == m2.p
    @test p == m3.p
    @test q == m1.q
    @test q == m2.q
    @test max(q, p-1) == m3.q

    # Check that model covariance matches
    c1 = model_covariance(m1, 15)
    c2 = model_covariance(m2, 15)
    c3 = model_covariance(m3, 15)
    c0 = c1[1]
    @test all(abs(c1-c2) .< EPSILON*c0)
    @test all(abs(c1-c3) .< EPSILON*c0)

    # Check that the initial covariances match
    # While this should be redundany with above test, let's just be sure
    NIV = max(0,q-p+1)
    if NIV>0
        @test all(abs(m1.covarIV[1:NIV].-m2.covarIV[1:NIV]) .< EPSILON*c0)
        @test all(abs(m1.covarIV[1:NIV].-m3.covarIV[1:NIV]) .< EPSILON*c0)
    end

    # Check that the model rational function representation matches.
    if m1.q > 0
        maxcoef = maximum(abs(m1.thetacoef))
        @test all(abs(m1.thetacoef.-m2.thetacoef) .< EPSILON*maxcoef)
        # At this point, the m3 theta polynomial is not at all guaranteed to match
        # the others, so omit that test for now. If the model_covariance matches,
        # this test is not critical, but we'll think over how it can be improved.
    end

    maxcoef = maximum(abs(m1.phicoef))
    @test all(abs(m1.phicoef.-m2.phicoef) .< EPSILON*maxcoef)
    @test all(abs(m1.phicoef.-m3.phicoef) .< EPSILON*maxcoef)
end


# 5) Test fitting data to a sum-of-exponentials representation
function test_sum_exp(bases::Vector, ampls::Vector, N::Integer)
    signal=zeros(Float64, N)
    for (b,a) in zip(bases,ampls)
        signal += real(a*(b.^(0:N-1)))
    end
    bfit,afit = fit_exponentials(signal, length(bases))
    # Rather than testing the fit, test the model that it generates.
    model=zeros(Float64, N)
    for (b,a) in zip(bfit,afit)
        model += real(a*(b.^(0:N-1)))
    end
    @test all(abs(model-signal) .< 1e-6)
end

bases=[.999,.98,.7+.1im,.7-.1im]
ampls=[5.0,4,3-1im,3+1im]
test_sum_exp(bases, ampls, 1000)

bases=[.99,.9,.1+.8im,.1-.8im]
ampls=[7.0,5,3-1im,3+1im]
test_sum_exp(bases, ampls, 1000)

bases=[.999, .99, .95, .9, .7]
ampls=[1,2,3,4,5]
test_sum_exp(bases, ampls, 1000)
