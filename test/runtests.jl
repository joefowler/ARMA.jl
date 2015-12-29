using ARMA
using Base.Test

# write your own tests here
@test 1 == 1

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

@test estimate_covariance([0,2,0,-2]) == [2.0, 0.0, -2.0]
u = randn(1026)
r = u[3:end] + u[1:end-2] + 2*u[2:end-1]
cv = estimate_covariance(r, 20)
@test abs(cv[1] - 6) < 1
@test abs(cv[2] - 4) < 1
@test abs(cv[3] - 1) < 1
@test abs(cv[4]) < 1
