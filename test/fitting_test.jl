using ARMA: innovations_estimate
using Test

@testset "innovations" begin
    t = 0:20
    r = (.99 .^t) .+ 10(.98 .^t) + 20(.9 .^t)
    for (p,q) in ((3,3), (3,5), (5,3))
        _,θ,ϕ = innovations_estimate(r, p, q)
        @test degree(θ) == q
        @test degree(ϕ) == p
        _,θ1,_ = innovations_estimate(r, p)
        @test degree(θ1) == p
    end
    # Can't make an ARMA(5,5) model from less than 11 data values.
    @test_throws BoundsError innovations_estimate(r[1:5], 5)
end
