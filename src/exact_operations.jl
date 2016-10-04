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
    for i=2:min(Nk,N)
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
        throw(DimensionMismatch("Second dimension of B, $(B.nrows), does not match length of v, $(length(v))"))
    end
    x = B.m[:,end] .* v
    for i=2:B.nbands
        x[i:end] += B.m[i:end,end+1-i] .* v[1:end+1-i]
    end
    x
end

function \(B::BandedLTMatrix, v::Vector)
    if B.nrows != length(v)
        throw(DimensionMismatch("Second dimension of B, $(B.nrows), does not match length of v, $(length(v))"))
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

function transpose_solve(B::BandedLTMatrix, v::Vector)
    if B.nrows != length(v)
        throw(DimensionMismatch("First dimension of B, $(B.nrows), does not match length of v, $(length(v))"))
    end
    x = similar(v)
    Nv = length(v)
    x[Nv] = v[Nv] / B[Nv,Nv]
    for i = Nv-1:-1:Nv-B.nbands+1
        x[i] = (v[i] - dot(vec(B[i+1:end,i]), x[i+1:end]))/B[i,i]
    end
    for i = Nv-B.nbands:-1:1
        x[i] = (v[i] - dot(vec(B[i+1:i+B.nbands,i]), x[i+1:i+B.nbands]))/B[i,i]
    end
    x
end

transpose_solve(M::Matrix, v::Vector) = M'\v

*(B::BandedLTMatrix, M::Matrix) = hcat([B*M[:,i] for i=1:size(M)[2]]...)
\(B::BandedLTMatrix, M::Matrix) = hcat([B\M[:,i] for i=1:size(M)[2]]...)


"
An object containing the information needed to perform certain noise-related operations
on a vector within an ARMA model, including the following (where `R` is the ARMA model
noise covariance matrix and `L` is its lower Cholesky factor):

1. L\\v  `whiten(solver, v)`
1. L*v  `unwhiten(solver, v)`
1. R\\v  `solve_covariance(solver, v)`
1. R*v  `mult_covariance(solver, v)`
"

type ARMASolver
    p         ::Int
    q         ::Int
    phicoef   ::Vector{Float64}
    covarIV   ::Vector{Float64}
    # RR = Phi*R*Phi', a banded matrix = RRt + RRu
    RRu       ::Matrix{Float64}   # The upper left (p)x(p+q) corner of RR-RRt
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
    covar = model_covariance(m, 1+m.p+2m.q)
    R_corner  = toeplitz(covar, covar)
    const Nc = length(covar)
    const Nbands = max(m.p, m.q+1)
    x = zeros(Float64, Nc)
    y = zeros(Float64, Nc)
    x[1:m.p+1] = m.phicoef
    y[1] = x[1]
    Phi = toeplitz(x, y)
    RR_corner = Phi*R_corner*Phi'
    RR_toeplitz = zeros(Float64, 2m.q+1)
    RR_toeplitz[1:m.q+1] = RR_corner[end, end-m.q:end]
    RR_toeplitz[m.q+1:end] = RR_corner[end, end:-1:end-m.q]
    RR_rectcorner = RR_corner[1:max(m.p,Nbands-1), :]

    # Now compute LL such that LL*LL' == RR, without ever representing RR as a full
    # matrix in memory. Awesome.
    LL = BandedLTMatrix(Float64, N, Nbands)
    LL_corner = chol(RR_corner, Val{:L})
    for r=1:min(Nc,N)
        for c=1+max(0,r-Nbands):r
            LL[r,c] = LL_corner[r,c]
        end
    end
    for r = Nc+1:N
        const mincol = r-m.q
        LL[r,mincol] = RR_toeplitz[1] / LL[mincol,mincol]
        for c=mincol+1:r-1
            LL[r,c] = (RR_toeplitz[c-r+Nbands] - dot(vec(LL[r,mincol:c-1]), vec(LL[c,mincol:c-1]))) / LL[c,c]
        end
        Rt = RR_toeplitz[m.q+1]
        S = sum( LL[ r,mincol:r-1] .^ 2)
        LL[r,r] = sqrt(Rt - S)
    end
    ARMASolver(m.p, m.q, m.phicoef, covar, RR_rectcorner, RR_toeplitz, LL)
end


"""`whiten(solver::ARMASolver, v::Vector)`

Use `solver` for an ARMA model to whiten the vector `v`. Here, "whiten" means
return `w=L\\v`, where `L` is the lower Cholesky factor of the covariance matrix.
The expected value of `w*w'` is the identity matrix."""

function whiten(solver::ARMASolver, v::Vector)
    Phiv = convolve_same(v, solver.phicoef)
    const nv = length(v)
    if nv  < solver.LL.nrows
        return solver.LL[1:nv, 1:nv] \ Phiv
    end
    solver.LL \ Phiv
end

function whiten(solver::ARMASolver, M::Matrix)
    ws(v::Vector) = whiten(solver, v)
    mapslices(ws, M, 1)
end




"""`unwhiten(solver::ARMASolver, w::Vector)`

Use `solver` for an ARMA model to unwhiten the vector `w`. Here, "unwhiten" means
return `v=L*w`, where `L` is the lower Cholesky factor of the covariance matrix.
If the expected value of `w*w'` is the identity matrix, then the expected
value of `v*v'` is the data covariance matrix of the ARMA model."""

function unwhiten(solver::ARMASolver, w::Vector)
    const nw = length(w)
    if nw  < solver.LL.nrows
        x = solver.LL[1:nw, 1:nw] * w
    else
        x = solver.LL * w
    end
    deconvolve_same(x, solver.phicoef)
end

function unwhiten(solver::ARMASolver, M::Matrix)
    uws(v::Vector) = unwhiten(solver, v)
    mapslices(ws, M, 1)
end


"""`mult_covariance(solver::ARMASolver, v::Vector)`

Use `solver` for an ARMA model to multiply the vector `v` by the covariance matrix."""

function mult_covariance(solver::ARMASolver, v::Vector)
    # Reversing before and after deconvolve ensures that we are solving Phi', not Phi.
    v1 = reverse(deconvolve_same(reverse(v), solver.phicoef))

    # Multiply by RR, the "moving-average-only R matrix".
    const Nv = length(v1)
    const Nu1,Nu2 = size(solver.RRu)
    const q = solver.q
    if Nv > Nu2
        x = similar(v1)
        x[1:Nu1] = solver.RRu * v1[1:Nu2]
        for r = Nu1+1 : Nv-q
            x[r] = dot(solver.RRt, v1[r-q:r+q])
        end
        for r = Nv-q+1 : Nv
            x[r] = dot(solver.RRt[1:end-r+Nv-q], v1[r-q:end])
        end

    elseif Nv > Nu1
        x = similar(v1)
        x[1:Nu1] = solver.RRu[:, 1:Nv] * v1
        for r = Nu1+1 : Nv-q
            x[r] = dot(solver.RRt, v1[r-q:r+q])
        end
        for r = max(Nu1,Nv-q)+1 : Nv
            x[r] = dot(solver.RRt[1:end-r+Nv-q], v1[r-q:end])
        end

    else
        x = solver.RRu[1:Nv,1:Nv] * v1
    end
    deconvolve_same(x, solver.phicoef)
end


function solve_covariance(solver::ARMASolver, v::Vector)
    const nv = length(v)
    v1 = whiten(solver, v)
    if nv  < solver.LL.nrows
        v2 = solver.LL[1:nv,1:nv]' \ v1
    else
        v2 = transpose_solve(solver.LL, v1)
    end
    reverse(convolve_same(reverse(v2), solver.phicoef))
end


function inverse_covariance(solver::ARMASolver, N::Integer)
    M = eye(N)
    hcat([solve_covariance(solver, M[:,i]) for i=1:N]...)
end
