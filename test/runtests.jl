using ARMA
using Base.Test

# padded_length: rounds up to convenient size for FFT

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

# estimate_covariance

@test estimate_covariance([0,2,0,-2]) == [2.0, 0.0, -2.0, 0.0]
u = randn(1026)
r = u[3:end] + u[1:end-2] + 2*u[2:end-1]
cv = estimate_covariance(r, 20)
@test abs(cv[1] - 6) < 1
@test abs(cv[2] - 4) < 1
@test abs(cv[3] - 1) < 1
for lag = 4:19
    @test abs(cv[lag]) < 1
end

@test estimate_covariance(u, 10) == estimate_covariance(u, 10, div(1026, div(1025+150,150)))
@test length(estimate_covariance(u)) == length(u)

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

# ARMAModel constructors
p,q = 3,3
rs = 1+(randn(q) .^ 2)
ps = 1+(randn(p) .^ 2)
sigma = 1.0
m = ARMAModel(sigma, rs, ps)
@test m.p == p
@test m.q == q
n = ARMAModel(m.thetacoef, m.phicoef)
@test m.thetacoef == n.thetacoef
@test m.phicoef == n.phicoef
@test similar_list(m.roots_, n.roots_, 1e-7)
@test similar_list(m.poles, n.poles, 1e-7)
