#
# Functions to work with exact multiplication and solutions for R and its Cholesky
# (lower) decomposition L
#

type ARMASolver
    p         ::Int
    q         ::Int
    thetacoef ::Vector{Float64}
    phicoef   ::Vector{Float64}
    covarIV   ::Vector{Float64}
    expbases  ::Vector{Complex128}
    expampls  ::Vector{Complex128}
    # RR = Phi*R*Phi', a banded matrix = RRt + RRu
    RRt       ::Vector{Float64}   # The banded Toeplitz part of RR
    RRu       ::Matrix{Float64}   # The upper left (p+1)x(p+1) corner of RR-RRt
    LL        ::Matrix{Float64}   # LL = lower Cholesky factor of RR, also banded

    function ARMASolver(p, q, thetacoef, phicoef, covarIV, expbases, expampls,
        RRt, RRu, LL)
        @assert p>=0
        @assert q>=0
        @assert p+1 == length(phicoef)
        @assert q+1 == length(thetacoef)
        @assert p == length(expbases)
        @assert p == length(expampls)
        @assert length(covarIV) >= 1+q
        @assert length(covarIV) >= p
        @assert thetacoef[1] > 0
        @assert phicoef[1] == 1.0

        new(p, q, thetacoef, phicoef, covarIV, expbases, expampls,
            RRt, RRu, LL)
    end
end

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

"Store a banded lower-triangular matrix"

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
    w::Int
end

function BandedLTMatrix{T}(m::Matrix{T})
    nrows, W = size(m)
    BandedLTMatrix(m, nrows, W)
end

import Base: getindex,size,*, \

function getindex(B::BandedLTMatrix, r::Integer, c::Integer)
    if r>B.nrows || c>B.nrows || r<=0 || c<=0
        throw(BoundsError(B,[r,c]))
    end
    if r<c || r-c >= B.w
        return 0
    end
    B.m[r, end+c-r]
end

size(B::BandedLTMatrix) = [B.nrows, B.nrows]

function *(B::BandedLTMatrix, v::Vector)
    if B.nrows != length(v)
        throw(DimensionMismatch("second dimension of B, $(B.nrows), does not match length of v, $(length(v))"))
    end
    x = B.m[:,end] .* v
    for i=2:B.w
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
    for i=2:B.w
        x[i] = (v[i] - dot(vec(B.m[i,end-i+1:end-1]), x[1:i-1]))/B[i,i]
    end
    for i=B.w+1:B.nrows
        x[i] = (v[i] - dot(vec(B.m[i,1:end-1]), x[i+1-B.w:i-1]))/B[i,i]
    end
    x
end
