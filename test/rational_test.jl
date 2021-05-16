using ARMA: BarycentricRational, roots_pfrac
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
    end
end
