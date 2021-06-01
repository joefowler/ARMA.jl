using ARMA
using LinearAlgebra
using Polynomials
using Test
using ToeplitzMatrices

@testset "ARMA" begin

# 1) Test padded_length: rounds up to convenient size for FFT
@testset "Padded length" begin
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
end

# 2) Test estimate_covariance
@testset "Estimate covariance" begin
    @test estimate_covariance([0,2,0,-2]) == [2.0, 0.0, -2.0, 0.0]
    u = randn(2+2^13)
    r = u[3:end] + u[1:end-2] + 2*u[2:end-1]
    cv = estimate_covariance(r, 20)
    @test abs.(cv[1] - 6) < 1
    @test abs.(cv[2] - 4) < 1
    @test abs.(cv[3] - 1) < 1
    for lag = 4:19
        @test abs.(cv[lag]) < 1
    end

    @test estimate_covariance(u, 10) == estimate_covariance(u, 10, div(length(u), div(length(u)-1+150,150)))
    @test length(estimate_covariance(u)) == length(u)
end

# 3) Basic tests of ARMAModel constructors
function similar_list(a::Vector, b::Vector, eps)
    @assert length(a) == length(b)
    for a1 in a
        if all(abs2.(b .- a1) .> eps^2)
            return false
        end
    end
    for b1 in b
        if all(abs2.(a .- b1) .> eps^2)
            return false
        end
    end
    true
end

@testset "ARMA constructors" begin
    p,q = 3,3
    rs = 1 .+ (randn(q) .^ 2)
    ps = 1 .+ (randn(p) .^ 2)
    variance = 1.0
    m = ARMAModel(rs, ps, variance)
    @test m.p == p
    @test m.q == q
    n = ARMAModel(m.θcoef, m.ϕcoef)
    @test m.θcoef == n.θcoef
    @test m.ϕcoef == n.ϕcoef
    @test similar_list(m.zroots, n.zroots, 1e-7)
    @test similar_list(m.zpoles, n.zpoles, 1e-7)
end

# 4) Now complete tests of several models that have been worked out carefully
# on paper, as well as several that are randomly created.

@testset "ARMA representations" begin
    # Generate 6 models of fixed parameters and order (2,0), (0,2), (1,1), (1,2), (2,1), (2,2)
    thetas=Dict('A'=>[2], 'B'=>[2,2.6,.8], 'C'=>[2,1.6], 'D'=>[2,2.6,.8], 'E'=>[2,1.6], 'F'=>[2,2.6,.8])
    phis = Dict('A'=>[1,-.3,-.4], 'B'=>[1], 'C'=>[1,-.8], 'D'=>[1,-.8], 'E'=>[1,-.3,-.4], 'F'=>[1,-.3,-.4])
    EPSILON = 2e-4

    # And generate 6 models of random order and random parameters
    for model in "GHIJKL"
        # Order will be 0<=p,q <=6.
        # Use rand^(-.3) for roots/poles. Negative power ensures abs.(r)>1, and
        # the 0.3 power concentrates the values near the unit circle.
        p = rand(0:6)
        q = rand(0:6)
        if p+q==0; p=q=5; end  # Don't test ARMA(0,0) model!
        zroots = rand(q) .^ (-.3)
        zpoles = rand(p) .^ (-.3)

        # Want one negative pole, if p>=3
        if p>2
            zpoles[end] *= -1
        end

        # On larger-order models, with probability 1/2 make one pair roots and/or poles complex.
        if p>2 && rand(0:1) == 1
            zpoles = complex(zpoles)
            zpoles[1] = complex(real(zpoles[1]),real(zpoles[2]))
            zpoles[2] = conj(zpoles[1])
        end

        if q>2 && rand(0:1) == 1
            zroots = complex(zroots)
            zroots[1] = complex(real(zroots[1]),real(zroots[2]))
            zroots[2] = conj(zroots[1])
        end

        # Scale theta by 0.7 to avoid lucky cancellations in the tests.
        thetas[model] = ARMA.polynomial_from_roots(zroots) * 0.7
        phis[model] = ARMA.polynomial_from_roots(zpoles)
        phis[model] *= 1.0/phis[model][1]
    end

    # Loop over all the models specified by their rational function representation
    # in thetas[] and phis[]. For each model, construct this way, then construct the
    # other 3 ways using the computed representations. Verify that the resulting
    # model has the same covariance and other key properties.

    for model in "ABCDEFGHIJKL"
        thcoef = thetas[model]
        phcoef = phis[model]
        if phcoef[1] != 1.0
            thcoef /= phcoef[1]
            phcoef /= phcoef[1]
        end
        @assert phcoef[1] == 1.0
        p = length(phcoef)-1
        q = length(thcoef)-1
        # println("Testing model $model of order ARMA($p,$q).")

        m1 = ARMAModel(thcoef, phcoef)

        roots_ = roots(Polynomial(thcoef))
        poles = roots(Polynomial(phcoef))
        expbases = 1.0 ./ poles

        # We'll be working with q+1 equations to find initial values
        # of psi: the Taylor expansion coefficients of θ(z)/ϕ(z).
        # See BD (3.3.3) for the q+1 initial equations and  (3.3.4) for the
        # homogeneous equations beyond the first q+1. Careful with the sign conventions,
        # b/c BD uses ϕ(z) = 1 - (ϕ1*z + ϕ2*z^2 + ...), while I prefer a + sign.
        phpad = fill(0.0, q+1)
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
            append!(phN, fill(0.0, q-p))
        end
        A = fill(0.0, N, N)
        for i=1:N
            for j=1:N
                col = 1+abs(j-i)
                A[i,col] += phN[j]
            end
        end
        rhs3_3_8 = fill(0.0, N)
        for k=1:q+1 # here j,k are both 1 larger than in BD 3.3.8.
            for j=k:q+1
                rhs3_3_8[k] += thcoef[j]*psi[1+j-k]
            end
        end
        gamma = A \ rhs3_3_8

        m2 = ARMAModel(roots_, poles, gamma[1])

        if q<p
            covarIV = Float64[]
        else
            covarIV = gamma[1:1+q-p]
        end
        B = Array{ComplexF64}(undef, p, p)
        for r=1:p
            for c=1:p
                B[r,c] = expbases[c]^(N-p+r-1)
            end
        end
        expampls = B \ gamma[N-p+1:N]

        # If p-q>1, then you can't really work from sum-of-exponentials to ARMA,
        # because you have to assume q=p-1, leading to infinte roots, etc etc.
        # As a hack, when this degenerate case is reached, skip the sum-exp representation.
        if p-q > 1
            m3 = m2
        else
            m3 = ARMAModel(expbases, expampls, covarIV)
        end

        # A) Check that model orders are equivalent
        # Take care with model m3, b/c it never sets q<p-1 when constructing.
        @test p == m1.p
        @test p == m2.p
        @test p == m3.p
        @test q == m1.q
        @test q == m2.q
        @test q == m3.q

        # B) Check that model covariance matches
        c1 = model_covariance(m1, 100)
        c2 = model_covariance(m2, 100)
        c3 = model_covariance(m3, 100)
        c0 = c1[1]
        @test all(abs.(c1 .- c2) .< EPSILON*c0)
        @test all(abs.(c1 .- c3) .< EPSILON*c0)

        # C) Check that the initial covariances match
        # While this should be redundant with above test, let's just be sure
        NIV = max(0,q-p+1)
        if NIV>0
            @test all(abs.(m1.covarIV[1:NIV].-m2.covarIV[1:NIV]) .< EPSILON*c0)
            @test all(abs.(m1.covarIV[1:NIV].-m3.covarIV[1:NIV]) .< EPSILON*c0)
        end

        # D) Check that the model rational function representation matches.
        if m1.q > 0
            maxcoef = maximum(abs.(m1.θcoef))
            @test all(abs.(m1.θcoef .- m2.θcoef) .< EPSILON*maxcoef)
            # At this point, the m3 θ polynomial is not at all guaranteed to match
            # the others, so omit that test for now. If the model_covariance matches,
            # this test is not critical, but we'll think over how it can be improved.
        end

        maxcoef = maximum(abs.(m1.ϕcoef))
        @test all(abs.(m1.ϕcoef.-m2.ϕcoef) .< EPSILON*maxcoef)
        @test all(abs.(m1.ϕcoef.-m3.ϕcoef) .< EPSILON*maxcoef)

        # E) Test model_psd. This isn't easy to see how to test, other than re-implement
        # the model_psd code itself!
        N = 50
        freq = collect(range(0, stop=0.5, length=N))
        z = exp.(-2im*pi *freq)
        numer = m1.θcoef[1] .+ fill(0.0im, N)
        for i=1:m1.q
            numer .+= m1.θcoef[i+1] * (z.^i)
        end
        denom = m1.ϕcoef[1] .+ fill(0.0im, N)
        for i=1:m1.p
            denom .+= m1.ϕcoef[i+1] * (z.^i)
        end
        psd = abs2.(numer ./ denom)
        threshold = 1e-3 * maximum(abs.(psd[1]))
        mpsd = model_psd(m1, N)
        @test size(psd) == size(mpsd)
        @test all(abs.(psd .- mpsd) .< threshold)
        # @test all(abs.(psd - model_psd(m2, N)) .< threshold)
        # @test all(abs.(psd - model_psd(m3, N)) .< threshold)

        mpsd = model_psd(m1, freq)
        @test size(psd) == size(mpsd)
        @test all(abs.(psd .- mpsd) .< threshold)
        # @test all(abs.(psd - model_psd(m2, freq)) .< threshold)
        # @test all(abs.(psd - model_psd(m3, freq)) .< threshold)

    end
end

# 5) Test fitting data to a sum-of-exponentials representation
# and an ARMA model of order (p, q=p)
function test_sum_exp(ampls::Vector, bases::Vector, N::Integer)

    # First, make sure that ARMA.exponential_model does the right thing.
    t = 0:(N-1)
    signal = ARMA.exponential_model(t, ampls, bases)
    signal2 = zero(signal)
    for (b,a) in zip(bases, ampls)
        signal2 .+= real(a*(b.^(0:N-1)))
    end
    @test all(abs.(signal2-signal) .< 1e-6*minimum(abs.(ampls)))

    # Now add a tiny bit of noise, fit exponentials and see what happens.
    # Rather than testing the fit, test the model that it generates.
    noise_level = 1e-4
    signal .+= randn(N)*noise_level
    NB = length(bases)
    afit, bfit = fit_exponentials(signal, pmin=NB, pmax=NB)
    cmodel = ARMA.exponential_model(t, afit, bfit)
    @test all(abs.(cmodel .- signal) .< .1*maximum(signal))

    # Now test the full fitARMA function, with 0 and then 1 exceptional value.
    p = length(bases)
    model = fitARMA(signal, p, p-1, deltar=noise_level, pmin=p-2)
    cmodel = model_covariance(model, N)
    @test all(abs.(cmodel .- signal) .< .1*maximum(signal))

    signal[1] *= 2
    model = fitARMA(signal, p, p, deltar=noise_level, pmin=p-2)
    cmodel = model_covariance(model, N)
    @test all(abs.(cmodel .- signal) .< .1*maximum(signal))
end

@testset "exponential_fits" begin
    ampls=[5.0,4,3-1im,3+1im]
    bases=[.999,.98,.7+.1im,.7-.1im]
    test_sum_exp(ampls, bases, 400)

    ampls=[7.0,5,3-1im,3+1im]
    bases=[.99,.9,.1+.8im,.1-.8im]
    test_sum_exp(ampls, bases, 400)

    ampls=[1,2,3,4,5]
    bases=[.999, .95, .9, .8, .5]
    test_sum_exp(ampls, bases, 400)
end

# 6) Test toeplitz_whiten and toeplitz_whiten! with an ARMA(2,2) and 5 random vectors
@testset "toeplitz_whiten" begin
    r=[3,-3]
    poles = [1.25,-2]
    model = ARMAModel(r, poles, 10.0)
    N = 50
    Phi = fill(0.0, N, N)
    for i=1:model.p+1
        for col=1:N+1-i
            Phi[col+i-1,col] = model.ϕcoef[i]
        end
    end
    Theta = fill(0.0, N, N)
    for i=1:model.q+1
        for col=1:N+1-i
            Theta[col+i-1,col] = model.θcoef[i]
        end
    end
    for i=1:5
        v = randn(N)
        correct_tw = Theta \ (Phi * v)
        tw = toeplitz_whiten(model, v)
        @test all(abs.(tw - correct_tw) .< 1e-6)
    end
end

arrays_similar(v::AbstractArray, w::AbstractArray, eps=1e-10) = all(abs.(v-w) .< eps)


# 7) Test internals used by whiten, unwhiten, solve_covariance, mult_covariance
@testset "whiten" begin
    for i=1:5
        N = 50
        v = randn(N)
        vx = copy(v)
        vx[2:end] += 0.8*v[1:end-1]
        vy = copy(v)
        vy[2:end] -= 0.3*v[1:end-1]
        vy[3:end] -= 0.4*v[1:end-2]

        @test arrays_similar( ARMA.convolve_same(v, [1, 0.8]), vx)
        @test arrays_similar( ARMA.deconvolve_same(vx, [1, 0.8]), v)
        @test arrays_similar( ARMA.convolve_same(v, [1, -.3, -.4]), vy)
        @test arrays_similar( ARMA.deconvolve_same(vy, [1, -.3, -.4]), v)
    end

    # Test whiten, unwhiten, solve_covariance, mult_covariance
    model23 = ARMAModel([1.2,1.1,1.02], [1.25, -2], 10)
    model32 = ARMAModel([1.25,-2], [1.2,1.1,1.02], 10)
    model52 = ARMAModel([1.25,-2], [6,2.5,1.2,1.1,1.02], 10)
    model25 = ARMAModel([6,2.5,1.2,1.1,1.02], [1.25, -2], 10)
    for model in (model23, model32, model25, model52)
        N = 16
        gamma = model_covariance(model, N)
        R = SymmetricToeplitz(gamma)
        L = cholesky(R).L
        x = fill(0.0, N)
        y = fill(0.0, N)
        x[1:model.p+1] = model.ϕcoef
        y[1] = x[1]
        Phi = Toeplitz(x, y)
        # must force symmetry, or numerical precision will make cholesky() fail.
        RR = Hermitian(Phi*R*Phi')
        LL = cholesky(RR).L

        solver = ARMASolver(model, N)
        for N in [2,4,6,8,10,14]
            v = randn(N)
            @test arrays_similar(LL[1:N,1:N]\v, solver.LL[1:N,1:N]\v, 1e-7)
            @test arrays_similar(LL[1:N,1:N]*v, solver.LL[1:N,1:N]*v, 1e-7)
            @test arrays_similar(L[1:N,1:N]\v, whiten(solver, v), 1e-6)
            @test arrays_similar(L[1:N,1:N]*v, unwhiten(solver, v), 1e-7)
            @test arrays_similar(R[1:N,1:N]*v, mult_covariance(solver, v), 1e-7)
            @test arrays_similar(R[1:N,1:N]\v, solve_covariance(solver, v), 1e-3)
            Rinv = inverse_covariance(solver, N)
            @test arrays_similar(R[1:N,1:N]*Rinv, Matrix{Float64}(I, N, N), 1e-7)
            @test arrays_similar(Rinv*R[1:N,1:N], Matrix{Float64}(I, N, N), 1e-7)

            # Test that they can be applied to matrices as well as vectors
            M = randn(N, 4)
            @test arrays_similar(L[1:N,1:N]\M, whiten(solver, M), 1e-5)
            @test arrays_similar(L[1:N,1:N]*M, unwhiten(solver, M), 1e-7)
            @test arrays_similar(R[1:N,1:N]*M, mult_covariance(solver, M), 1e-7)
            @test arrays_similar(R[1:N,1:N]\M, solve_covariance(solver, M), 1e-3)
        end

    end
end

include("hdf5test.jl")
include("rcproots_test.jl")
include("rational_test.jl")
include("fitting_test.jl")
end
