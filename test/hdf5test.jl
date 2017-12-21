using Base.Test
using ARMA
using HDF5

function compare(m1::ARMAModel, m2::ARMAModel)
    @test m1.p == m2.p
    @test m1.q == m2.q
    @test m1.covarIV ≈ m2.covarIV atol=3e-4 * (2^m1.p)
    @test model_covariance(m1, 50) ≈ model_covariance(m2, 50) atol=3e-4 * (2^m1.p)
end

@testset "hdf5 save/load" begin
    ntests = 5
    for i = 1:ntests
        nbases = rand(2:6)
        bases = rand(nbases)
        ampls = 10*rand(nbases)
        covarIV = [3*sum(abs.(ampls))]
        model = ARMAModel(bases, ampls, covarIV)

        fname1 = tempname()*".hdf5"
        ARMA.hdf5save(fname1, model)
        model1 = ARMA.hdf5load(fname1)
        compare(model, model1)

        fname2 = tempname()*".hdf5"
        h5open(fname2, "w") do f
            g1 = g_create(f, "top")
            g2 = g_create(g1, "mid")
            g3 = g_create(g2, "low")
            ARMA.hdf5save(g3, model)
        end
        h5open(fname2, "r") do f
            model2 = ARMA.hdf5load(f["top/mid/low"])
            compare(model, model2)
        end
    end
end #testset
