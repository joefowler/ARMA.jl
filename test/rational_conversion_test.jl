using ARMA: RealRational, PartialFracRational, PairedPartialFracRational

@testset "Rational Conversions" begin
    allroots1 = [1+.5im, 1-.5im, 2, 3, .5+1im, .5-1im, 3]
    allpoles1 = [.5+1.2im, .5-1.2im, 1.2+.5im, 1.2-.5im, -1+.5im, -1-.5im, 4]
    allroots2 = [2.2, -2.5, 3.3, 3.8, -2.1, 2.5, 10]
    allpoles2 = allroots2 .- 0.1
    ztest1 = LinRange(-1.9, 1.9, 501)
    ztest2 = exp.(1im*LinRange(0, π, 501))
    ztest = vcat(ztest1, ztest2)

    for (allroots, allpoles) in zip((allroots1, allroots2,allpoles2), (allpoles1, allpoles2,allroots2))
        # Test both with and without remainder polynomial, and with and without extra poles.
        # (2,6), (3,7), (3,5) have extra poles (4, 4, and 1, respectively).
        # (2,6) and (4,4) have constant remainder;
        # (3,7), (3,5) and (3,4) have zero remainder; (6,4) has quadratic remainder
        for (q,p) in ((2,6), (3,6), (3,7), (3,4), (4,4), (6,4))
            r = allroots[1:q]
            λ = allpoles[1:p]
            f0 = 4.0

            rr1 = RealRational(r, λ, f0)
            pfr1 = PartialFracRational(rr1)
            rr2 = RealRational(pfr1)

            ppfr = PairedPartialFracRational(pfr1)
            pfr2 = PartialFracRational(ppfr)
            rr3 = RealRational(pfr2)

            for x in r
                @test minimum(abs.(x.-rr2.zroots)) < 1e-8
                @test minimum(abs.(x.-rr3.zroots)) < 1e-8
            end
            for x in λ
                @test minimum(abs.(x.-rr2.zpoles)) < 1e-8
                @test minimum(abs.(x.-rr3.zpoles)) < 1e-8
            end
            @test all(isapprox.(rr1.ϕ, rr2.ϕ; atol=1e-10))
            @test all(isapprox.(rr1.ϕ, rr3.ϕ; atol=1e-10))

            # Check that the function evaluation is equal
            truef = rr1(ztest)
            for equivalent in (pfr1, rr2, ppfr, pfr2, rr3)
                @test all(isapprox.(truef, equivalent(ztest); atol=1e-10, rtol=1e-10))
            end
        end
    end

    # Test a case where conversion to PFR requires re-sorting poles.
    # Consider a (q,p)=(2,7) rational. It will put 3 poles into the partial fraction, and 4
    # will be "extra factors".
    λ = [1+1im, 1-1im, 1.2+1im, 1.2-1im, 2+.5im, 2-.5im, -3]
    r = [3, 4.0]
    f0 = 4.0

    rr1 = RealRational(r, λ, f0)
    pfr1 = PartialFracRational(rr1)
    rr2 = RealRational(pfr1)
end
