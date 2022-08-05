using ARMA: PartialFracRational, RealRational

@testset "Rational Conversions" begin
    allroots = [1+.5im, 1-.5im, 2, 3, .5+1im, .5-1im]
    allpoles = [.5+1.2im, .5-1.2im, 1.2+.5im, 1.2-.5im, -1+.5im, -1-.5im]
    for (q,p) in ((2,6), (3,4), (4,4), (6,4))
        r = allroots[1:q]
        λ = allpoles[1:p]
        f0 = 4.0

        rr = RealRational(r, λ, f0)
        @show q, p, rr
        pfr = PartialFracRational(rr)
        rr2 = RealRational(pfr)
        for x in r
            @test min(abs.(x.-rr2.zroots)) < 1e-8
        end
        for x in λ
            @test min(abs.(x.-rr2.zpoles)) < 1e-8
        end
        @test maximum(rr.ϕ-rr2.ϕ) < 1e-8
        @test minimum(rr.ϕ-rr2.ϕ) > -1e-8

        # ppfr = PairedPartialFracRational(pfr)
        # pfr2 = PartialFracRational(ppfr)
        # rr2 = RealRational(pfr2)
    end
end
