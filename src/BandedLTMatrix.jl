# Note: March 21, 2017
#
# We are no longer using a BandedLTMatrix object to represent a banded lower-triangular
# matrix, because it was slower than using a SparseMatrixCSC by a factor of at least
# 2-3 on all operations, including construction, M-v multiply, M-M multiply,
# M-v solve, M-M solve, and transpose operations.  Keep the code in git for now,
# but I think that writing our own matrix type was too specialized to do efficiently.
# Lesson learned.

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

# Performance comment (March 16, 2017)
# It seems that we would achieve faster multiplication and solves for BandedLTMatrix
# if we were to replace it with a sparse type (e.g., speye(N)). The construction of
# the BandedLTMatrix is MUCH faster, typically 50x. But the mulitply and solve operations
# are approximately 2x faster for the sparse version. Thus, it might be smart to
# remove this type entirely and replace its use with SparseMatrixCSC, someday...

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
    x = similar(v)
    for i=1:N
        x[i] = v[i]*B.m[i,end]
    end
    const Nb = B.nbands
    for i=2:Nb
        for k=i:N
            x[k] += B.m[k, Nb+1-i] * v[k-i+1]
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
        d = v[i]
        for k=i+1:Nv
            d -= B[k,i] * x[k]
        end
        x[i] = d/B[i,i]
    end
    for i = Nv-B.nbands:-1:1
        d = v[i]
        for k=i+1:i+B.nbands
            d -= B[k,i]*x[k]
        end
        x[i] = d/B[i,i]
    end
    x
end

function transpose_solve(B::BandedLTMatrix, M::AbstractMatrix)
    if B.nrows != size(M)[1]
        throw(DimensionMismatch("first dimension of B, $(B.nrows), does not match length of M, $(size(M)[1])"))
    end
    x = similar(M)
    Nv,Nc = size(M)
    for c=1:Nc
        x[Nv,c] = M[Nv,c] / B[Nv,Nv]

        for i = Nv-1:-1:Nv-B.nbands+1
            d = M[i,c]
            for k=i+1:Nv
                d -= B[k,i] * x[k,c]
            end
            x[i,c] = d/B[i,i]
        end
        for i = Nv-B.nbands:-1:1
            d = M[i,c]
            for k=i+1:i+B.nbands
                d -= B[k,i]*x[k,c]
            end
            x[i,c] = d/B[i,i]
        end
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

function \(B::BandedLTMatrix, M::AbstractMatrix)
    R = similar(M)
    for c=1:size(M)[2]
        R[:,c] = B\M[:,c]
    end
    R
end


# Here are two tests that used to live in runtests.jl.
# But we aren't using BandedLTMatrix, so they've been removed.
#
function testBandedLT()
    S = 6
    b = BandedLTMatrix(Int, S, S-1)
    for i=1:S
        for j=1:S
            @test b[i,j] == 0
        end
    end
    @test_throws ArgumentError b=BandedLTMatrix(Int, 3, 6)
    @test_throws ArgumentError b=BandedLTMatrix(eye(4), 0) # Caught in constructor
    @test_throws ErrorException b=BandedLTMatrix(eye(4), -3) # Errors before constructor

    for nbands = 1:6
        bx = eye(6); bx[nbands,1] = 4
        # Check that band-counting works
        b = BandedLTMatrix(bx)
        @test b.nbands == nbands

        # Check that band-counting allows values < eps to count as zero
        b = BandedLTMatrix(bx; eps=10)
        @test b.nbands == 1

        # Check that we can insist on a # of bands regardless of the input m.
        forced_nb = 3
        b = BandedLTMatrix(bx, forced_nb)
        @test b.nbands == forced_nb
        if nbands > forced_nb
            @test b[nbands, 1] == 0
        end
    end

    # Be sure that the zero parts of the array are of the right type
    b1 = BandedLTMatrix(eye(S))
    b2 = BandedLTMatrix(eye(S), 2)
    @test typeof(b1[1,1]) == Float64
    @test typeof(b1[S,1]) == Float64
    @test typeof(b1[1,S]) == Float64
    @test typeof(b2[1,1]) == Float64
    @test typeof(b2[S,1]) == Float64
    @test typeof(b2[1,S]) == Float64

    # Test B*v, B\v, B*M, and B\M for vector v and matrix M
    v = randn(S)
    M = randn(S,3)
    b = eye(S)*3 # Let b be diagonally dominant
    for r=2:S
        b[r,r-1:r] += randn(2)
    end
    B = BandedLTMatrix(b)
    @test B.nbands == 2
    @test arrays_similar(B*v, b*v)
    @test arrays_similar(B*M, b*M)
    @test arrays_similar(B\v, b\v)
    @test arrays_similar(B\M, b\M)
    @test arrays_similar(ARMA.transpose_solve(B, v), b'\v)
    @test arrays_similar(ARMA.transpose_solve(B, v), B'\v)
end

function test7_whiten_internals()
    for i=1:5
        N = 50
        v = randn(N)
        vx = copy(v)
        vx[2:end] += 0.8*v[1:end-1]
        vy = copy(v)
        vy[2:end] -= 0.3*v[1:end-1]
        vy[3:end] -= 0.4*v[1:end-2]

        @test arrays_similar( ARMA.convolve_same(v, [1, 0.8]), vx)
        @test arrays_similar( ARMA.deconvolve_same(vx, [1, 0.8]), v)
        @test arrays_similar( ARMA.convolve_same(v, [1, -.3, -.4]), vy)
        @test arrays_similar( ARMA.deconvolve_same(vy, [1, -.3, -.4]), v)
    end

    for j=1:5
        N, Nb = 30, 4
        B = ARMA.BandedLTMatrix(randn(N,N), Nb)
        B.m[:,end] += 2  # Make B diagonally dominant
        M = zeros(Float64, N, N)
        for i=1:Nb
            M += diagm(B.m[i:end, end+1-i],  1-i)
        end
        for i=1:5
            v = randn(N)
            @test arrays_similar(M*v, B*v)
            @test arrays_similar(M\v, B\v)
            X1 = M'\v
            X2 = ARMA.transpose_solve(B, v)
            @test arrays_similar(X1, X2)
        end
    end
end
