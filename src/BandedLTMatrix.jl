"""`BandedLTMatrix(T, nrows, nbands)`

`BandedLTMatrix(m; eps=0.0)`

`BandedLTMatrix(m, nbands)`

Store a square, banded lower-triangular matrix in space proportional to
`nrows*nbands` instead of `nrows^2`. Also perform fast operations that make use
of this sparsity. So far only multiplication by a vector and solving a vector
(operators *,\) are needed, so nothing else has been implemented.

The first form creates a matrix with `zero(T)` for all values, and with `nrows`
rows and columns. Only the diagonal and the first `(nbands-1)` sub-diagonals can
hold non-zero values. An ArgumentError is thrown when `nrows < nbands`.

The second form `b=BandedLTMatrix(m::AbstractMatrix; eps=0)` stores matrix `m`
into a new object `b` (ignoring the upper triangle of `m`). It infers the
number of bands by checking bands from furthest below the diagonal, moving
towards the diagonal until a band with at least one non-zero value is found. By
default, the test is for strict equality. If you set  the keyword argument
`eps`, then the test will be for absolute values less than `eps`.

The third form `b=BandedLTMatrix(m::AbstractMatrix, nbands::Integer)` stores
matrix `m` into a new object `b` like the second form, except that `nbands`
is given by the caller instead of being detected by searching for nonzero values.
Any values in `m` above the diagonal or on or below the `nbands`-th sub-diagonal
are ignored.
"""

# The internal representation is
#[.   .   .   M11
# .   .   M21 M22
# .   M31 M32 M33
# M41 M42 M43 M44 etc]
# where the . means that the value is never used.
#
# This representation puts the diagonal in m[:,end], the first sub-diagonal in
# m[2:end, end-1] and so on.

type BandedLTMatrix{T <: Number} <: AbstractMatrix{T}
    nrows::Int
    nbands::Int
    zeroval::T
    m::Matrix{T}

    function BandedLTMatrix{T}(nrows::Int, nbands::Int, m::Matrix{T})
        nr,nc = size(m)
        if nr != nrows
            throw(ArgumentError("a BandedLTMatrix requires nrows==size(m)[1], but nrows=$nrows and size(m)==($nr,$nc)"))
        elseif nrows < 1
            throw(ArgumentError("a BandedLTMatrix requires nrows>0, but nrows=$nrows"))
        elseif nbands < 1
            throw(ArgumentError("a BandedLTMatrix requires nbands>0, but nbands=$nbands"))
        elseif nbands > nrows
            throw(ArgumentError("a BandedLTMatrix requires nrows ≥ nbands, but $(nrows) < $(nbands)"))
        end
        new(nrows, nbands, zero(T), m)
    end
end
BandedLTMatrix{T <: Number}(nrows::Int, nbands::Int, m::AbstractMatrix{T}) = BandedLTMatrix{T}(nrows, nbands, m)

# Construct by (Type, #rows, #bands)
BandedLTMatrix(T::Type, nr, nb) = BandedLTMatrix{T}(nr, nb, zeros(T, nr, nb))

# Construct by (input; zero-threshold)
function BandedLTMatrix{T <: Number}(m::AbstractMatrix{T}; eps::Real=0.0)
    if eps<0
        throw(ArgumentError("a BandedLTMatrix(m, eps=ϵ) requires ϵ ≥ 0, but eps=$(eps)"))
    end
    nrows, nbands = size(m)
    for nbands = nrows:-1:2
        for r = nbands:nrows
            if abs(m[r, r+1-nbands]) > eps
                return BandedLTMatrix(m, nbands)
            end
        end
    end
    BandedLTMatrix(m, 1)
end

# Construct by (input, #bands)
function BandedLTMatrix{T <: Number}(m::AbstractMatrix{T}, nbands)
    nrows, ncols = size(m)
    if nrows != ncols
        throw(ArgumentError("a BandedLTMatrix must be square, but size=($nrows,$ncols"))
    end
    b = BandedLTMatrix(T, nrows, nbands)
    for r = 1:nbands
        b[r, 1:r] = m[r, 1:r]
    end
    for r = nbands+1:nrows
        b[r, r+1-nbands:r] = m[r, r+1-nbands:r]
    end
    b
end


Base.linearindexing{T<:BandedLTMatrix}(::Type{T}) = Base.LinearFast()

import Base: getindex,setindex!,size,*, \

function getindex(B::BandedLTMatrix, r::Integer, c::Integer)
    if r>B.nrows || c>B.nrows || r<=0 || c<=0
        throw(BoundsError(B,[r,c]))
    end
    if r<c || r-c >= B.nbands
        return B.zeroval
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

size(B::BandedLTMatrix) = (B.nrows, B.nrows)

function *(B::BandedLTMatrix, v::Vector)
    const N = length(v)
    if B.nrows != N
        throw(DimensionMismatch("second dimension of B, $(B.nrows), does not match length of v, $(N)"))
    end
    x = copy(v)
    x .*= B.m[:,end]
    for i=2:B.nbands
        for k=i:N
            x[k] += B.m[k, end+1-i] * v[k-i+1]
        end
    end
    x
end

function \(B::BandedLTMatrix, v::Vector)
    const N = length(v)
    if B.nrows != N
        throw(DimensionMismatch("second dimension of B, $(B.nrows), does not match length of v, $(N)"))
    end
    x = similar(v)
    x[1] = v[1] / B[1,1]
    for i=2:B.nbands
        d = v[i]
        for k=1:i-1
            d -= B.m[i, B.nbands-i+k] * x[k]
        end
        x[i] = d/B[i,i]
    end
    for i=B.nbands+1:B.nrows
        d = v[i]
        for k=1:B.nbands-1
            d -= B.m[i,k] * x[i+k-B.nbands]
        end
        x[i] = d/B[i,i]
    end
    x
end

function transpose_solve(B::BandedLTMatrix, v::Vector)
    if B.nrows != length(v)
        throw(DimensionMismatch("first dimension of B, $(B.nrows), does not match length of v, $(length(v))"))
    end
    x = similar(v)
    Nv = length(v)
    x[Nv] = v[Nv] / B[Nv,Nv]

    for i = Nv-1:-1:Nv-B.nbands+1
        x[i] = (v[i] - dot(vec(B[i+1:Nv,i]), x[i+1:end]))/B[i,i]
    end
    for i = Nv-B.nbands:-1:1
        x[i] = (v[i] - dot(vec(B[i+1:i+B.nbands,i]), x[i+1:i+B.nbands]))/B[i,i]
    end
    x
end

transpose_solve(M::Matrix, v::Vector) = M'\v

function *(B::BandedLTMatrix, M::AbstractMatrix)
    s1,s2 = size(M)
    if B.nrows != s1
        throw(DimensionMismatch("second dimension of B, $(B.nrows), does not match length of M, $(s1)"))
    end
    x = zeros(M)
    for i=1:B.nbands
        for j=1:s2
            for k=i:s1
                x[k,j] += B.m[k,end+1-i] * M[k-i+1,j]
            end
        end
    end
    x
end

function \(B::BandedLTMatrix, M::Matrix)
    R = similar(M)
    for c=1:size(M)[2]
        R[:,c] = B\M[:,c]
    end
    R
end
