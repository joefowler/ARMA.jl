using ARMA: BarycentricRational, PartialFracRational, roots_pfrac, aaawt
using Test

@testset "BarycentricRational" begin
    @testset "roots_pfrac functions" begin
        for testnum = 1:5
            n = rand(1:8)
            w = randn(n)
            x = randn(n)*100
            f(z) = sum(w./(z.-x))
            for s in (0, 1, π, 1e4)
                r = roots_pfrac(w, x, s)
                diff = f.(r) .- s
                @test all((abs.(diff) .< 1e-8*max(s, 1)))
            end
        end
    end

    @testset "BR functions" begin
        for testnum = 1:5
            n = rand(1:8)
            λ = randn(n)*100
            f = randn(n)*100
            w = rand(n)
            br = BarycentricRational(λ, f, w)
            @test all(br.(λ) .≈ f)
            @test br(Inf) ≈ sum(f.*w)/sum(w)
            testx = LinRange(minimum(λ), maximum(λ), 12)
            testf = br(testx)
            @test all(isfinite.(testf))
        end
        @test_throws DimensionMismatch p=BarycentricRational([1,2], [3,4,5,6,7], [0,2])
        @test_throws ArgumentError p=BarycentricRational([1,2], [3,4], [0,0])
    end
end

@testset "AAA rational approx" begin
    f(x) = 1.0/(x-2)+x^2
    z = collect(LinRange(-1, 1, 51))
    w = ones(Float64, length(z))
    a1 = aaawt(z, f.(z), 3)
    a2 = aaawt(z, f, 3)
    a3 = aaawt(z, f.(z), w, 3)
    a4 = aaawt(z, f, w, 3)
    fapprox = a1(z)
    @test all(a2(z) ≈ fapprox)
    @test all(a3(z) ≈ fapprox)
    @test all(a4(z) ≈ fapprox)
    @test a1(Inf) == a1(-Inf)
end

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
    @test all((p4(z).-f1 .== 1) .| isinf.(f1))
    @test all((f1 .< p5(z)) .| isinf.(f1))

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
    expect .+= 1.0 .+ 2z .+ 3*(1.5z.^2 .- 0.5)
    @test all(expect .≈ p(z))
end
