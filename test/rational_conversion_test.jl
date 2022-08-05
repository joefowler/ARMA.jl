using ARMA: PartialFracRational, RealRational

@testset "Rational Conversions" begin
    allroots = [1+.5im, 1-.5im, 2, 3, .5+1im, .5-1im]
    allpoles = [.5+1.2im, .5-1.2im, 1.2+.5im, 1.2-.5im, -1+.5im, -1-.5im]
    for (q,p) in ((2,6), (3,4), (4,4), (6,4))
        r = allroots[1:q]
        λ = allpoles[1:p]
        f0 = 4.0

        rr1 = RealRational(r, λ, f0)
        pfr1 = PartialFracRational(rr1)
        rr2 = RealRational(pfr1)
        for x in r
            @test minimum(abs.(x.-rr2.zroots)) < 1e-8
        end
        for x in λ
            @test minimum(abs.(x.-rr2.zpoles)) < 1e-8
        end
        @test all(isapprox.(rr1.ϕ, rr2.ϕ; atol=1e-10))

        # ppfr = PairedPartialFracRational(pfr1)
        # pfr2 = PartialFracRational(ppfr)
        # rr3 = RealRational(pfr2)
    end
end
