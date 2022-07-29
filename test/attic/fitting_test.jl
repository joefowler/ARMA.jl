using ARMA: partial_frac_decomp, make_poles_legal, roots, vectorfit, RCPRoots, PartialFracRational
using Test

@testset "vectorfit" begin
    N = 101
    p = 4
    z = LinRange(-1, 1, N)
    z0 = .5+.1im
    σ = 0.1  # noise level
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

@testset "make_poles_legal" begin
    poles = [[-1.5, 45], [1.5, .5], [1.0, 3]]
    mustchange = [false, true, true]
    z = LinRange(-1, .9, 25)
    for (p, expect) in zip(poles, mustchange)
        pf = PartialFracRational(p, [6., -12], [1.])
        PSD = pf(z) .+ randn(length(z))
        pfl = make_poles_legal(pf, z, PSD)
        @test xor(all(pf.λ .≈ pfl.λ), expect)
    end
    pf = PartialFracRational([.98, 1.0], [6., -12], [1.])
    PSD = pf(z) .+ randn(length(z))
    @test_throws DimensionMismatch make_poles_legal(pf, z, PSD[1:end-1])
    @test_throws DimensionMismatch make_poles_legal(pf, z, PSD, ones(Float64, length(z)+5))
    pfl = make_poles_legal(pf, z, PSD)
    @test all(abs2.(pfl.λ .- 1) .< 5e-8)
end
