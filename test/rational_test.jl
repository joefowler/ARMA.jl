using ARMA: BarycentricRational, roots_pfrac, aaawt
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
