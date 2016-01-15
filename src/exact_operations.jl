#
# Functions to work with exact multiplication and solutions for R and its Cholesky
# (lower) decomposition L
#

"Convolve the vector `b` with `kernel`, yielding a vector of the same size
as `b`, effectively padding it with zeros as needed. This is equivalent to
multiplying `b` by the banded lower triangular Toeplitz matrix whose first
row begins with `kernel`."

function convolve_same(b::Vector, kernel::Vector)
    x = kernel[1] * copy(b)
    for i=2:length(kernel)
        x[i:end] += kernel[i] * b[1:end+1-i]
    end
    x
end

"Reverse the effect of `convolve_same(b, kernel)`. That is, solve Kx=b for
x where K is the banded lower triangular Toeplitz matrix whose first row
begins with `kernel`."
function deconvolve_same(b::Vector, kernel::Vector)
    const Nk, N = length(kernel), length(b)
    lenrek = reverse(kernel[2:end])
    x = copy(b) / kernel[1]
    for i=2:Nk
        x[i] = (x[i]-dot(kernel[i:-1:2], x[1:i-1])) / kernel[1]
    end
    for i=Nk+1:N
        x[i] = (x[i]-dot(lenrek, x[i-Nk+1:i-1])) / kernel[1]
    end
    x
end

"Store a banded lower-triangular matrix.
So far only multiplication by a vector and solving a vector (operators *,\)
are needed, so that's all that's implemented."

# Representation is
#[.   .   .   M11
# .   .   M21 M22
# .   M31 M32 M33
# M41 M42 M43 M44 etc]
# where the . means that the value is never used.
# This representation puts the diagonal in m[:,end], the first
# sub-diagonal in m[2:end, end-1] and so on.

type BandedLTMatrix{T} <: AbstractMatrix{T}
    m::Matrix{T}
    nrows::Int
    nbands::Int
end

function BandedLTMatrix{T}(m::Matrix{T})
    nrows, nbands = size(m)
    BandedLTMatrix(m, nrows, nbands)
end

# BandedLTMatrix{T}(nr::Integer, nb::Integer) = BandedLTMatrix{T}(Array{T}(nr, nb))
BandedLTMatrix(T::Type, nr::Integer, nb::Integer) = BandedLTMatrix(zeros(T, nr, nb))

import Base: getindex,setindex!,size,*, \

function getindex(B::BandedLTMatrix, r::Integer, c::Integer)
    if r>B.nrows || c>B.nrows || r<=0 || c<=0
        throw(BoundsError(B,[r,c]))
    end
    if r<c || r-c >= B.nbands
        return 0
    end
    B.m[r, end+c-r]
end

function setindex!(B::BandedLTMatrix, x, r::Integer, c::Integer)
    if r>B.nrows || c>B.nrows || r<=0 || c<=0
        throw(BoundsError(B,[r,c]))
    end
    if x != 0 && (r<c || r-c >= B.nbands)
        throw(BoundsError(B,[r,c]))
    end
    B.m[r, end+c-r] = x
end

size(B::BandedLTMatrix) = [B.nrows, B.nrows]

function *(B::BandedLTMatrix, v::Vector)
    if B.nrows != length(v)
        throw(DimensionMismatch("second dimension of B, $(B.nrows), does not match length of v, $(length(v))"))
    end
    x = B.m[:,end] .* v
    for i=2:B.nbands
        x[i:end] += B.m[i:end,end+1-i] .* v[1:end+1-i]
    end
    x
end

function \(B::BandedLTMatrix, v::Vector)
    if B.nrows != length(v)
        throw(DimensionMismatch("second dimension of B, $(B.nrows), does not match length of v, $(length(v))"))
    end
    x = similar(v)
    x[1] = v[1] / B[1,1]
    for i=2:B.nbands
        x[i] = (v[i] - dot(vec(B.m[i,end-i+1:end-1]), x[1:i-1]))/B[i,i]
    end
    for i=B.nbands+1:B.nrows
        x[i] = (v[i] - dot(vec(B.m[i,1:end-1]), x[i+1-B.nbands:i-1]))/B[i,i]
    end
    x
end

*(B::BandedLTMatrix, M::Matrix) = hcat([B*M[:,i] for i=1:size(M)[2]]...)
\(B::BandedLTMatrix, M::Matrix) = hcat([B\M[:,i] for i=1:size(M)[2]]...)

type ARMASolver
    p         ::Int
    q         ::Int
    phicoef   ::Vector{Float64}
    covarIV   ::Vector{Float64}
    # RR = Phi*R*Phi', a banded matrix = RRt + RRu
    RRu       ::Matrix{Float64}   # The upper left (p+1)x(p+1) corner of RR-RRt
    RRt       ::Vector{Float64}   # The banded Toeplitz part of RR
    LL        ::BandedLTMatrix{Float64}   # LL = lower Cholesky factor of RR, also banded

    function ARMASolver(p, q, phicoef, covarIV, RRu, RRt, LL)
        @assert p>=0
        @assert q>=0
        @assert p+1 == length(phicoef)
        @assert length(covarIV) >= 1+q
        @assert length(covarIV) >= p
        @assert phicoef[1] == 1.0

        new(p, q, phicoef, covarIV, RRu, RRt, LL)
    end
end


function ARMASolver(m::ARMAModel, N::Integer)
    covar = model_covariance(m, max(m.p, m.q+1))
    R_corner  = toeplitz(covar, covar)
    Nc = length(covar)
    x = zeros(Float64, Nc)
    y = zeros(Float64, Nc)
    x[1:m.p+1] = m.phicoef
    y[1] = x[1]
    Phi = toeplitz(x, y)
    RR_corner = Phi*R_corner*Phi'
    RR_toeplitz = vec(RR_corner[end, end-Nc+1:end])
    # Now compute LL such that LL*LL' == RR, without ever representing RR as a full
    # matrix in memory. Awesome.
    LL = BandedLTMatrix(Float64, N, length(covar))
    LL_corner = chol(RR_corner, Val{:L})
    for r=1:Nc
        for c=1:r
            LL[r,c] = LL_corner[r,c]
        end
    end
    for r = Nc+1:N
        const mincol = r-Nc+1
        LL[r,mincol] = RR_toeplitz[1] / LL[mincol,mincol]
        for c=r-Nc+2:r-1
            LL[r,c] = (RR_toeplitz[c-r+Nc] - dot(vec(LL[r,mincol:c-1]), vec(LL[c,mincol:c-1]))) / LL[c,c]
        end
        LL[r,r] = sqrt(RR_toeplitz[Nc] - sum( LL[r,mincol:r-1] .^ 2))
    end
    ARMASolver(m.p, m.q, m.phicoef, covar, RR_corner, RR_toeplitz, LL)
end


function whiten(solver::ARMASolver, v::Vector)
    Phiv = convolve_same(v, solver.phicoef)
    solver.LL \ Phiv
end

function unwhiten(solver::ARMASolver, v::Vector)
    v
end
