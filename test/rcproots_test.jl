using ARMA: RCPRoots, nreal, ncomplex, realroots
using Test

@testset "RCPRoots" begin
    @testset "real values" begin
        rireal = [1, 2, -3]
        rreal = [1., 3, 4.]
        ricplx = [1, 2, 3, 4+0im]
        rcplx = [1.0, 2, 3, 4+0im]
        for x in (rireal, rreal, ricplx, rcplx)
            r = RCPRoots(x)
            L = length(x)
            @test nreal(r) == L && ncomplex(r) == 0 && length(r) == L
            @test eltype(r) <: Real
            @test all(x .== r)
            @test all(real(r) .== real(x))
            @test all(imag(r) .== 0)
        end
    end

    @testset "complex values" begin
        c1 = [1, 2, 3+1im, 3-1im]
        c2 = [1.0, 2, 3+1im, 3-1im]
        c3 = [1.0, 2-π*1im, 3, 2+π*1im]
        c4 = [1.0+1e-15im, 2, 3+4im, 3-4im]
        for x in (c1, c2, c3, c4)
            r = RCPRoots(x)
            @test nreal(r) == 2 && ncomplex(r) == 2 && length(x) == 4
            @test r[1] == conj(r[2]) && imag(r[1]) > 0
            @test all([conj(z) in r for z in r])
            @test length(realroots(r)) == 2
        end
    end

    @testset "errors" begin
        bad1 = [1, 2, 3+1im] # not paired
        bad2 = [1, 2, 3+1im, 4-1im] # bad pair
        bad3 = [1, 2, 3+1im, 3-1im-1e-10im]  # bad pair (at default abstol and reltol)
        bad4 = [1, 2, 1, 1+1e-10im]  # bad pair (at default angletol)
        for x in (bad1, bad2, bad3, bad4)
            @test_throws DomainError r = RCPRoots(x)
        end
        r = RCPRoots(bad3; reltol=1e-7, abstol=1e-7)
        @test nreal(r) == 2
        r = RCPRoots(bad4; angletol=1e-7)
        @test nreal(r) == 4
    end
end
