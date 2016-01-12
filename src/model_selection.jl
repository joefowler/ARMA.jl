#
# Functions to find the appropriate ARMA model from data.
#

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

"""Find the `nexp` "main exponentials" in the time stream `data`.

Specifically, follow the prescription of XXX & YYY in "blah" *J. Statistics* (1995),
building a matrix H whose columns are contiguous segments of the time stream.
Perform a singular value decomposition to find the rank-`nexp` decomposition of H,
from which we can construct the square matrix A (of size `nexp`) which is known
to be similar to the "system matrix" (i.e., a time-step-advance matrix). The
latter has eigenvalues equal to the exponentials comprising the time stream, so
A (by similarity) has the same eigenvalues.

That's the best explanation I have. It might be more honest to say that we
find the exponentials by magic.

Note that the current implementation does not construct the exact SVD of H,
but rather uses a randomized matrix technique to compute an *approximate* SVD
very efficiently, containing only a specified number of leading singular vectors.

TODO: we need a better heuristic for constructing H out of a sufficient number
of segments from the data, without allowing it to get insanely large when `data`
is very, very long.
"""

function main_exponentials(data::Vector, nexp::Int)
    N = length(data)
    ncol = 10 + 2*nexp
    H = zeros(Float64, N+1-ncol, ncol)
    for c=1:ncol
        H[:,c] = data[c:c+N-ncol]
    end
    U,s,V = find_svd_randomly(H[1:end-1,:], nexp)
    W = diagm(1.0 ./ sqrt(s))
    A = W*U'*H[2:end,:]*V*W
    eigvals(A)
end


"Fit the time series `data` as a sum of `nexp` possibly complex exponentials.
Use `main_exponentials` to determine the exponential bases, then perform a
simple least-squares fit to determine the amplitudes."

function fit_exponentials(data::Vector, nexp::Int)
    bases = main_exponentials(data, nexp)
    N = length(data)
    M = Array{eltype(bases)}(N, nexp)
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

# Absent further information, we find that q=p is a good choice most of the time,
# so let that be a default:
fitARMA(covariance::Vector, p::Int) = fitARMA(covariance, p, p)

# Now if the user has no idea what model order to use, that's a much more
# complicated story.
function fitARMA(covariance::Vector)
    error("We don't have a plan yet for fitting ARMA models without a given order.")
end
