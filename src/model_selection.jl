include("RandomMatrix.jl")


function padded_length(N::Integer)
    powerof2 = round(Int, 2^ceil(log2(N)))
    if N == powerof2
        return N
    elseif N >= 0.75 * powerof2
        return powerof2
    elseif N >= 0.625 * powerof2
        return div(3*powerof2, 4)
    else
        return div(5*powerof2, 8)
    end
end


"""estimate_covariance(timeseries::Vector, nsamp::Int, chunklength::Int)

estimate_covariance takes a noise sequences (as a 1D vector
or as columns of a 2D matrix) and estimates the sample-sample covariance
vector from these.

nsamp and chunklength are both optional arguments.

nsamp is the number of samples to compute from the covariance. If not positive,
then compute all possible values.

chunklength is the amount of data to be considered at once. If zero, then this
will be either the length of the timeseries or 15*nsamp, whichever is shorter.

The idea is that a full computation is very inefficient. Making chunklength
much smaller than the timeseries length allows computation more efficient, but
at the cost of losing a small amount of correlation between values in successive
chunks. The recommended value for long data sets is to let chunklength be at
least 10-20x nsamp.
"""

estimate_covariance(timeseries::Vector) = estimate_covariance(timeseries, length(timeseries))

function estimate_covariance(timeseries::Vector, nsamp::Int)
    const N = length(timeseries)
    ideal_chunksize = 15*nsamp
    nchunks = div(N-1+ideal_chunksize, ideal_chunksize)
    chunklength = div(N, nchunks)
    estimate_covariance(timeseries, nsamp, chunklength)
end

function estimate_covariance(timeseries::Vector, nsamp::Int, chunklength::Int)
    const N = length(timeseries)
    if nsamp > N
        error("Cannot compute $(nsamp) covariance values from a length-$N data.")
    end
    paddedsize = padded_length(chunklength+nsamp)
    padded_data = zeros(Float64, paddedsize)
    result = zeros(Float64, nsamp)
    datamean = mean(timeseries)

    i=0
    chunks_consumed = 0
    while i+chunklength <= N
        padded_data[1:chunklength] = timeseries[i+1:i+chunklength] - datamean
        chunks_consumed += 1

        power = abs2(rfft(padded_data))
        acsum = irfft(power, length(padded_data))
        result += acsum[1:nsamp]
        i += chunklength
    end
    result ./ (chunks_consumed * collect(chunklength:-1:chunklength+1-nsamp))
end


function main_exponentials(data::Vector, nexp::Int)
    N = length(data)
    ncol = 10 + 2*nexp
    h = zeros(Float64, N+1-ncol, ncol)
    for c=1:ncol
        h[:,c] = data[c:c+N-ncol]
    end
    U,s,V = find_svd_randomly(h[1:end-1,:], nexp)
    W = diagm(1.0 ./ sqrt(s))
    A = W*U'*h[2:end,:]*V*W
    @show s, W, A
    eigvals(A)
end


function fit_exponentials(data::Vector, nexp::Int)
    bases = main_exponentials(data, nexp)
    # TODO: Fit for their amplitudes
    N = length(data)
    M = Array{Float64}(N, nexp)
    for i=1:nexp
        M[:,i] = bases[i] .^ (0:N-1)
    end
    amplitudes = pinv(M)*data
    bases, amplitudes
end

function fitARMA(covariance::Vector, p::Int, q::Int)
    nspecial = max(1+q-p, 0)
    bases, amplitudes = fit_exponentials(covariance[1+nspecial:end], p)
    amplitudes ./= (bases .^ nspecial)
    ARMAModel(bases, amplitudes, covariance[1:nspecial])
end
