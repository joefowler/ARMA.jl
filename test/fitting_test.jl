using ARMA: vectorfit, RCPRoots
using Test

@testset "vectorfit" begin
    N = 101
    p = 4
    z = LinRange(-1, 1, N)
    z0 = .5+.1im
    σ = 0.2
    f = σ*randn(N).+20 .- 1.0 ./ (z.-1.1) .+ 1.0 ./ (z.+1.1) .+ real(1.0 ./ (z.+z0) .+ 1.0 ./ (z.+conj(z0)))
    wt = ones(N)
    λ0 = complex(randn(p))
    λr = RCPRoots(λ0)
    vf1 = vectorfit(z, f, wt, λ0)
    vf2 = vectorfit(z, f, λ0)
    vf3 = vectorfit(z, f, wt, λr)
    vf4 = vectorfit(z, f, λr)
    for vf in (vf2, vf3, vf4)
        @test all(vf1(z) ≈ vf(z))
    end
    @test maximum(abs.(f.-vf1(z))) < 10σ

    @test vectorfit(z, f, Float64[]) != nothing
    @test_throws DimensionMismatch vectorfit(z, f[1:end-1], λ0)
    @test_throws DimensionMismatch vectorfit(z, f, ones(Float64, N-5), λ0)
    @test_throws ErrorException vectorfit(z, f, λ0, length(λ0)-2)
end
