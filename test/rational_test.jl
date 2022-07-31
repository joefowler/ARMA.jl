using ARMA: PartialFracRational, partial_frac_decomp, roots_pfrac
using Polynomials
using Test

@testset "PartialFracRational" begin
    @test_throws AssertionError p=PartialFracRational([1,2],[3,4]; polyMin=1, polyMax=1)
    @test_throws DimensionMismatch p=PartialFracRational([1,2],[3,4,5,6,7])
    @test_throws MethodError p=PartialFracRational()

    # Check that no b, b=[], and b=[0] are all equivalent
    z = LinRange(-2, 2, 93)
    p1 = PartialFracRational([1,2], [3,4.0])
    p2 = PartialFracRational([1,2], [3,4.0], [])
    p3 = PartialFracRational([1,2], [3,4.0], [0])
    f1 = p1(z)
    @test all(f1 .≈ p2(z))
    @test all(f1 .≈ p3(z))

    # Check that b=[1] and b=[1,0.4] are greater than the same Pfrac with b=[0]
    p4 = PartialFracRational([1,2], [3,4.0], [1])
    p5 = PartialFracRational([1,2], [3,4.0], [1, .4])
    @test all((p4(z).-f1 .≈ 1) .| isinf.(f1) .| isnan.(f1))
    @test all((f1 .< p5(z)) .| isinf.(f1))

    # Check derivatives, order 0-3.
    d0 = derivative(p4, 0)
    d1 = derivative(p4)  # check implied order=1
    d2 = derivative(p4, 2)
    d3 = derivative(p4, 3)
    p4d1 = -3 ./(z.-1).^2 .- 4 ./(z.-2).^2
    p4d2 = +6 ./(z.-1).^3 .+ 8 ./(z.-2).^3
    p4d3 = -18 ./(z.-1).^4 .- 24 ./(z.-2).^4
    @test all(d0.(z) .≈ p4(z))
    @test all(d1.(z) .≈ p4d1)
    @test all(d2.(z) .≈ p4d2)
    @test all(d3.(z) .≈ p4d3)

    # Check it behaves with implicit or explicit zero partial fraction terms.
    p1 = PartialFracRational(Float64[], Float64[], [13,4.0])
    p2 = PartialFracRational([13,4.0])
    f2(x) = 13.0+4x
    p3 = PartialFracRational([13,4.0]; polyMin=2, polyMax=4)
    f3(x) = 13.0+4(x-3)
    @test all(p1(z) .≈ p2(z))
    @test all(p2(z) .≈ f2.(z))
    @test all(p3(z) .≈ f3.(z))

    # Check a simple case, without and with remainder polynomial
    p = PartialFracRational([1,3,5], [4,5,6])
    z = collect(0:.1:4)
    expect = 4 ./(z.-1) .+ 5 ./(z.-3) .+ 6 ./(z.-5)
    @test all(expect .≈ p(z))

    p = PartialFracRational([1,3,5], [4,5,6], [1, 2, 3])
    expect .+= 1.0 .+ 2z .+ 3*(2z.^2 .- 1)
    @test all(expect .≈ p(z))
end

@testset "partial_frac" begin
    # Rewrite a/∏(z-λ[i]) as ∑ b[i]/z-λ[i]
    λ = [1, 2+1im, 2-1im]
    a=4
    b1 = partial_frac_decomp(a, λ)
    b8 = partial_frac_decomp(8a, λ)
    f = PartialFracRational(λ, b1)
    f8 = PartialFracRational(λ, b8)
    g(x) = a/prod(x.-λ)

    xtest = LinRange(1.1, 5, 60)
    @test all(f(xtest) .≈ g.(xtest))
    @test all(f8(xtest) .≈ 8g.(xtest))
end

@testset "PFR_roots" begin
    # Check that roots_pfrac0 works for real or complex values
    for r in ([1,2], [1+2im,4-2im])
        p = [3,4+1im,4+3im]
        weights = [
            (p[1]-r[1])*(p[1]-r[2])/(p[1]-p[2])/(p[1]-p[3]),
            (p[2]-r[1])*(p[2]-r[2])/(p[2]-p[1])/(p[2]-p[3]),
            (p[3]-r[1])*(p[3]-r[2])/(p[3]-p[1])/(p[3]-p[2])
        ]
        rs = ARMA.roots_pfrac0(weights, p)
        @test isapprox(sum([weights[i]./(rs[1].-p[i]) for i=1:3]), 0; atol=1e-11)
        @test isapprox(sum([weights[i]./(rs[2].-p[i]) for i=1:3]), 0; atol=1e-11)
    end


    p1 = PartialFracRational([-1, -2], [3, 2])
    p2 = PartialFracRational([-1, -2], [6, -12], [1])
    p3 = PartialFracRational([-1, -2], [-6, 24], [-6, 1])
    p4 = PartialFracRational([-1, -2], [-6, 24], [-5, 1, 2])
    pfr = [p1, p2, p3, p4]
    answers = [[-1.6], [1, 2], [0, 1, 2], nothing]
    for (p, answer) in zip(pfr, answers)
        r = roots(p)
        @test all(abs.(p.(r)) .< 1e-12)
        if answer != nothing
            rr = real(r)
            sort!(rr)
            @test all(isapprox.(rr, answer; atol=1e-10))
        end
    end

    function mindist(r::Vector)
        md = Inf
        for i=1:length(r)
            for j=1:i-1
                md = min(md, abs(r[i]-r[j]))
            end
        end
        md
    end

    testpoles = [1+1im,1-1im,.8+.7im,.8-.7im,1.1,1.01,-1.5,1.03]
    testresidues = [1.1+.1im,1.1-.1im,1.2+.2im,1.2-.2im,1.04,-1.5,2,1.4]
    remainder = [1,2,3,4,5,6,7,8.]
    for p=4:8
        for q=3:8
            m = min(p, q+1)
            # Follow the rule that even p, even q, and q<p need a constant remainder to keep from
            # splitting a complex pair of poles. (This way, either the partial fractions or the
            # extra factors of 1/(x-λ) use an EVEN number of poles.)
            if (p%2==0) && (q%2==0) && (m<p)
                m -= 1
            end

            pfr = PartialFracRational(testpoles[1:p], testresidues[1:m],remainder[1:q-m+1])
            @test pfr.p == p
            @test pfr.q == q
            @test pfr.m == m
            # Make sure all roots are distinct to the 1e-5 level and that the function
            # value is always small.
            for method in (:Eig, :Poly, :Both)
                r = roots(pfr; method=method)
                @test length(r) ==  q
                @test mindist(r) > 1e-5
                @test all(abs.(pfr(r)) .< 1e-7)
            end
        end
    end

    # The above suite of tests should be adequate, but why not keep these legacy tests in place, too?
    pole = ComplexF64[0.9999138564455727 + 1.311948345590019e-5im, 0.9999138564455727 - 1.311948345590047e-5im, 1.0000039082557004, 1.0001814813594927]
    residue = ComplexF64[-0.009648614144046466 - 0.015909481835920475im, -0.009648614144046797 + 0.01590948183592052im, 0.010087771996966484, -0.6895883924969154]
    remainder = [52.25867169674221, 34.27540419493968, -0.8505261672687883, 4.1627407018820985, -9.675698272165642]
    pfr = PartialFracRational(pole, residue, remainder)
    r = roots(pfr)

    # Make sure all roots are distinct to the 1e-5 level and that the function
    # value is always small.
    @test mindist(r) > 1e-5
    @test all(abs.(pfr(r)) .< 1e-7)

    pole = ComplexF64[0.9996695605301016 + 7.87923300847346e-5im, 0.9996695605301016 - 7.87923300847346e-5im, 1.0000039439142947, 1.0001291013807305]
    residue = ComplexF64[-0.004387340627692793 + 0.006942158051290787im, -0.004387340627692793 - 0.006942158051290787im, 0.012338398116147082, -0.5283333626371072]
    remainder = [79.58549226977715, 61.245019632968, -12.497377376570174]
    pfr = PartialFracRational(pole, residue, remainder)
    r = roots(pfr)

    @test mindist(r) > 1e-5
    @test all(abs.(pfr(r)) .< 1e-7)
end
