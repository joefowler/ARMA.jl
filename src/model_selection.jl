#
# Functions to find the appropriate ARMA model from data.
#

include("RandomMatrix.jl")
include("optimize_exponentials.jl")

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


"""
    estimate_covariance(timeseries::Vector, nsamp::Int, chunklength::Int)

Take a noise sequence (as a 1D vector or as columns of a 2D matrix) and
estimate the sample-sample covariance vector from these.

`nsamp` and `chunklength` are both optional arguments.

`nsamp` is the number of samples to compute from the covariance. If not a positive
number, then compute all possible values.

`chunklength` is the amount of data to be considered at once. If omitted, then this
will be either the length of the timeseries or 15*nsamp, whichever is shorter.

The idea is that a full computation is very inefficient. Making chunklength
much smaller than the timeseries length allows computation more efficient, but
at the cost of losing a small amount of correlation between values in successive
chunks. The recommended value for long data sets is to let chunklength be at
least 10-20x nsamp.
"""
estimate_covariance(timeseries::AbstractVector) = estimate_covariance(timeseries, length(timeseries))

function estimate_covariance(timeseries::AbstractVector, nsamp::Int)
    N = length(timeseries)
    ideal_chunksize = 15*nsamp
    nchunks = div(N-1+ideal_chunksize, ideal_chunksize)
    chunklength = div(N, nchunks)
    estimate_covariance(timeseries, nsamp, chunklength)
end

function estimate_covariance(timeseries::Vector, nsamp::Int, chunklength::Int)
    N = length(timeseries)
    if nsamp > N
        error("Cannot compute $(nsamp) covariance values from a length-$N data.")
    end
    paddedsize = padded_length(chunklength+nsamp)
    padded_data = zeros(Float64, paddedsize)
    result = zeros(Float64, nsamp)

    i=0
    chunks_consumed = 0
    while i+chunklength <= N
        datamean = mean(timeseries[i+1:i+chunklength])
        padded_data[1:chunklength] = timeseries[i+1:i+chunklength] - datamean
        chunks_consumed += 1

        power = abs2.(rfft(padded_data))
        acsum = irfft(power, length(padded_data))
        result += acsum[1:nsamp]
        i += chunklength
    end
    result ./ (chunks_consumed * collect(chunklength:-1:chunklength+1-nsamp))
end



"""
    main_exponentials(data, nexp; minexp=nothing)

Find the `nexp` "main exponentials" in the time stream `data`.

Specifically, follow the prescription of P. De Groen & B. De Moor (1987).
"The fit of a sum of exponentials to noisy data." J. Computational & Applied
Mathematics, vol. 20, pages 175–187.

We build a Hankel matrix H whose columns are contiguous segments of the time stream.
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

If `minexp==nothing` (the default), then only the length-`nexp` model is found.
If `minexp ≤ nexp`, then this function returns an array of arrays, one for each
value `p` with `minexp ≤ p ≤ nexp`, so that the length of the first is `minexp`
and the length of the last is `nexp`.

TODO: we need a better heuristic for constructing H out of a sufficient number
of segments from the data, without allowing it to get insanely large when `data`
is very, very long.
"""
function main_exponentials(data::Vector, nexp::Int; minexp=nothing)
    N = length(data)
    if 2nexp > N
        error("Cannot compute $(nexp) exponentials from data with fewer than twice as many data values.")
    end
    if (minexp != nothing) && (minexp > nexp)
        error("The value of `minexp` must be nothing or a number no larger than `nexp`.")
    end

    ncol = min(40 + 5nexp, div(N, 2))
    H = zeros(Float64, N+1-ncol, ncol)
    for c=1:ncol
        H[:,c] = data[c:c+N-ncol]
    end
    U,s,V = find_svd_randomly(H[1:end-1,:], nexp)
    W = diagm(s .^ (-0.5))
    A = W*U'*H[2:end,:]*V*W

    # By default, return the exponential set of size nexp only.
    if minexp == nothing
        return sortbases!(eigvals(A))
    end

    # The minexp has been set, so return all sizes minexp...nexp, inclusive.
    exps = []
    for p = minexp:nexp
        A = W[1:p, 1:p]*U[:,1:p]'*H[2:end,:]*V[:,1:p]*W[1:p,1:p]
        push!(exps, sortbases!(eigvals(A)))
    end
    exps
end


"""
    sortbases!(b)

Re-order the elements of `b` in place, such that for every pair b[1],b[2] and so
on, either both are real, or they are complex conjugates of one another.
Elements of `b` with negative imaginary parts are simply assumed to be
conjugates of those with positive imaginary parts. The complex pairs will come
first, followed by the real elements.

Returns the sorted `b`.
"""
function sortbases!(b::Vector)
    real_ones = real(b[imag(b) .== 0])
    sort!(real_ones, rev=true)
    if length(real_ones) < length(b)
        imag_ones = b[imag(b) .> 0]
        if length(imag_ones) != sum(imag(b) .< 0)
            error("sortbases!(b) requires that b contain equal numbers of elements with imag(x) > 0 and < 0.")
        end
        for i=1:length(imag_ones)
            b[2i-1] = imag_ones[i]
            b[2i] = conj(imag_ones[i])
        end
        b[2*length(imag_ones)+1:end] = real_ones[:]
    else
        b[:] = real_ones[:]
    end
    b
end


"""
    B2C(B)

Converts an array `B` of exponential bases to the coefficients of the set of
monic quadratic polynomials.

It is assumed that B are arranged such that B[1] and B[2] are either both real
or are complex conjugates of one another, and similarly for elements (3,4), (5,6),
and so on. Use `sortbases!(B)` to achieve this.

If `B` violates these assumptions, then the computed `C` will make no sense.

Returns `C` such that B[1:2] are the roots of x^2+C[1]*X+C[2], and similarly for
[3:4] and all other pairs. If length(B) is odd, then B[end] is the root of x+C[end],
thus `-1 = B[end]*C[end]`.
"""
function B2C{T<:Number}(B::AbstractVector{T})
    C = zeros(Float64, length(B))
    n = length(B)
    for i=1:2:n-1
        C[i] = -real(B[i]+B[i+1])
        C[i+1] = real(B[i]*B[i+1])
    end
    if n%2 > 0
        C[end] = -1/real(B[end])
    end
    C
end



"""
    C2B(C)

Converts a real array of polynomial coefficients `C` to their roots. See `B2C(B)`
for more.

Returns `B`, the possibly complex roots.
"""
function C2B{T<:Real}(C::AbstractVector{T})
    B = zeros(Complex{eltype(C)}, length(C))
    n = length(C)
    iscomplex = false
    for i = 1:2:n-1
        x = -0.5C[i]
        disc = x*x-C[i+1]
        if disc >= 0
            y::ComplexF64 = sqrt(disc)
        else
            y = 1im*sqrt(-disc)
            iscomplex = true
        end
        B[i] = x+y
        B[i+1] = x-y
    end
    if n%2 > 0
        B[end] = -1/C[end]
    end
    if !iscomplex
        return real(B)
    end
    B
end



"""
    findA(t, r, B, [w=weights])

Compute the amplitudes `A` for a linear model of the form

r_t ≈ f(t) = Sum_{i=1...p} A[i] B[i]^t

by minimizing

Sum_t [w_t (r_t - f(t))^2].

This is a linear problem, given `B`. If `w` is not given, then equal weights on
all data are assumed.

Returns the array of amplitudes `A`.
"""
function findA{T<:Number}(t::AbstractVector, r::Vector, B::Vector{T}; w=nothing)
    @assert length(t) == length(r)
    if w==nothing
        w = ones(r)
    end
    wr = w.*r
    p = length(B)
    M = zeros(T, p, p)
    D = zeros(T, p)
    for i=1:p
        for j=1:i
            M[j,i] = M[i,j] = sum(w .* (B[i]*B[j]).^t)
        end
        D[i] = sum(wr .* B[i].^t)
    end

    # A = M\D is theoretically correct, but don't do this: can be ill-conditioned in certain cases
    A = pinv(M) * D
end



"""
    exponential_model(t, A, B)`

Computes and returns the sum-of-exponentials model with amplitudes `A` and
exponential bases `B` at time steps `t`.
"""
function exponential_model(t::AbstractVector, A::Vector, B::Vector)
    r = zeros(Float64, length(t))
    for i=1:length(A)
        r += real(A[i]* B[i].^t)
    end
    r
end


"""
    squashgrowingexp!(bases::Vector, epsfrom1)

Replace any growing exponentials, that is, where  `abs(bases[j]) >= 1`, with
random decaying exponentials. These will be between `1-epsfrom1` and 1 in absolute
value (suggest `epsfrom1 <= 0.01`).
"""
function squashgrowingexp!(bases::Vector, epsfrom1::Float64)
    lastscale = 1 - 0.5*epsfrom1
    for (i,b) = enumerate(bases)
        if abs(b) >= 1 - 0.01*epsfrom1
            if i%2==1 || imag(b) == 0.0
                lastscale = (1-rand(1)[1]*epsfrom1) / abs(b)
            end
            bases[i] *= lastscale
        end
    end
end

"""
    fit_exponentials(data; pmin=0, pmax=6, w=nothing, deltar=nothing, good_enough=0.0)

Fit the time series `data` (regularly sampled) as a sum of possibly complex
exponentials numbering at least `pmin` and at most `pmax`. Uses
`main_exponentials` to determine a starting guess for the exponential bases.
Then a nonlinear fit is attempted, using `NLopt` with the `:LD_MMA` method.

Returns `A,B`, the amplitudes and exponential bases of the best-fit model. The
length of either gives the order of the best-fit model.

# Optional Arguments:
The penalty function is a weighted sum-of-square-errors. If the optional
argument `w` is given, it must be a vector of the same length as `data`, and
then the penalty is the sum over all samples of `w[i]*(model[i]-data[i])^2`.

If `w` is absent, then the weights are taken to be all equal to `deltar^(-2)`.

If `deltar` is also not given, then it is estimated from the standard deviation
of the successive-differences from the last 1/4 of the samples in `data`, on the
assumption that these are noise-dominated. If they aren't, then the caller must
offer a better figure for `w` or `deltar`.

If `pmin` < `pmax`, then models of more than one order will be tried, and the one
with the lowest penalty function will be returned.

If `good_enough` is assigned a positive value, and if any model order `p<pmax`
produces a penalty function less than this value, then this function returns
early with the lowest-order model that meets the criterion.

See also: [`fitARMA`](@ref)
"""
function fit_exponentials(data::Vector; pmin::Int=0, pmax::Int=6,
    w=nothing, deltar=nothing, good_enough::Float64=0.0)

    guess_exponentials = main_exponentials(data, pmax, minexp=pmin)

    N = length(data)
    if w == nothing
        # If deltar isn't given, use the heuristic that the std dev of the
        # diff of the last values in data estimates sqrt(2) times deltar.
        if deltar == nothing
            n = div(N, 4)
            deltar = std(diff(data[end-n:end]), mean=0) / sqrt(2)
        end

        w = ones(Float64, N) / float(deltar^2)
    end

    best_cost = +Inf
    best_fit = nothing
    for (guessB, p) = zip(guess_exponentials, pmin:pmax)
        squashgrowingexp!(guessB, 0.25/N)
        guessC = B2C(guessB)
        cost, fitC = optimize_exponentials(data, w, guessC)
        if (cost < best_cost)
            best_cost = cost
            best_fit = fitC
            # Stop early if cost is small enough.
            if cost <= good_enough
                break
            end
        end
    end

    bestB = C2B(best_fit)
    squashgrowingexp!(bestB, 0.25/N)

    t = 0:(length(data)-1)
    bestA = findA(t, data, bestB, w=w)
    bestA, bestB
end


"""
    fitARMA(covariance, p, q; kwargs)

Fit an `ARMAModel` of order `(p,q)` or lower to the `covariance` data. The model
order with the lowest cost function will be returned.

# Optional Arguments
- `pmin`: the minimum allowed ARMA order.
- `w`: the vector of weights to apply to the data (default: nothing, meaning equal weights)
- `deltar`: rms uncertainty on the covariance values.
- `good_enough`: if given, then the lowest-order model with a cost function at least
   this small will be used, rather than trying all orders up to `p`.

See also: [`fit_exponentials`](@ref)
"""
function fitARMA(covariance::Vector, p::Int, q::Int;
    w=nothing, deltar=nothing, good_enough::Float64=0.0, pmin::Int=0)
    nspecial = max(1+q-p, 0)
    amplitudes, bases = fit_exponentials(covariance[1+nspecial:end], pmin=pmin, pmax=p,
        w=w, deltar=deltar, good_enough=good_enough)
    if nspecial > 0
        amplitudes ./= (bases .^ nspecial)
    end
    ARMAModel(bases, amplitudes, covariance[1:nspecial])
end

# Absent further information, we find that q=p is a good choice most of the time,
# so let that be a default. Also, if we need a maximum order, we can supply
# the rather arbitrary guess of 6.
fitARMA(covariance::Vector, p::Int; kwargs...) = fitARMA(covariance, p, p; kwargs...)
fitARMA(covariance::Vector; kwargs...) = fitARMA(covariance, 6; kwargs...)
