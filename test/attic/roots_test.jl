using ARMA: legendre_roots, chebyshev_roots

@testset "Chebyshev roots" begin
    for testnum=1:5
        deg = rand(2:8)
        coef = randn(deg)
        # function L(x) f=coef[1]; for i=2:deg; f+= coef[i]*legendre(x, i-1); end; f; end
        # r = legendre_roots(coef)
        # @test all(abs.(L.(r)) .< 1e-8)

        CT = ChebyshevT(coef)
        C(x) = evalpoly(x, CT, false)  # allow out-of-domain evaluation
        r = chebyshev_roots(coef)
        @test all(abs.(C.(r)) .< 1e-6)
    end
end
