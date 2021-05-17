# A weighted version of the AAA algorithm for rational approximation.
#
# Based on Nakatsukasa, Y., Sète, O., & Trefethen, L. N. (2018). "The AAA Algorithm for Rational Approximation."
# _SIAM Journal on Scientific Computing_, 40(3), A1494–A1522. https://doi.org/10.1137/16M1106122
# but with an optional weighting vector.
#
# AAA is an algorithm that takes a large vector of sample points and function values there and approximates
# them by a rational function of a given order. AAA is a greedy algorithm. One by
# one, it decides which single point is in the poorest agreement with the model and should therefore be
# moved to the "support set" that will be exactly interpolated. All other
# points will be approximated; the approximation parameters are chosen by a least-squares fit.
#
# When a non-uniform weighting vector is given, the algorithm changes in two places:
# 1) Points are moved to the support set based on their _weighted_ difference from the model so far.
# 2) The free parameters are chosen by a _weighted_ least-squares fit to the non-support points and values.



# Handle function inputs as well as the default vector inputs.
function aaawt(z::AbstractVector{T}, f::S, w::AbstractVector{U}, n::Integer; tol=1e-13, verbose=false) where {S<:Function, T, U}
    aaawt(z, f.(z), w, n; tol=tol, verbose=verbose)
end
function aaawt(z::AbstractVector{T}, f::S, n::Integer; tol=1e-13, verbose=false) where {S<:Function, T}
    aaawt(z, f.(z), ones(T, length(z)), n; tol=tol, verbose=verbose)
end
function aaawt(z::AbstractVector{T}, f::AbstractVector{S}, n::Integer; tol=1e-13, verbose=false) where {S, T}
    aaawt(z, f, ones(T, length(z)), n; tol=tol, verbose=verbose)
end


"""
    aaawt(z, f, [w,] n; <keyword arguments>)

Return a rational function approximation for f(z) given a set of (f,z) data and weights w at each point.

Output is the AAA approximant as a callable `BarycentricRational` struct with fields (`z, f, w`),
vectors of support pts, function values, weights. Based on a paper by Nakatsukasa, Y., Sète, O., & Trefethen,
L. N. (2018), "The AAA Algorithm for Rational Approximation."
_SIAM Journal on Scientific Computing_, 40(3), A1494–A1522. https://doi.org/10.1137/16M1106122
but with an optional weighting vector.

# Arguments
- `z`: vector of sample points
- `f`: vector of data values at the points in z (must be same length as z); alternately, `f` can
  be a callable function, in which case `f.(z)` will be used as data values
- `w`: vector of weights to assign the model at each point; if omitted, equal weight will be used for each point
- `n`: (maximum) order of the rational approximation. If the tolerance can be met, the
   return value might be of order lower than `n`.
# Keyword arguments
- `tol=1e-13`: relative tolerance for model-data agreement
- `verbose=false` print info while calculating


"""
function aaawt(z::AbstractVector{T}, f::AbstractVector{S}, w::AbstractVector{U}, n::Integer; tol=1e-13,
             verbose=false) where {S, T, U}
    # filter out any NaN's or Inf's in the input
    keep = isfinite.(f) .& (w .> 0)
    f = f[keep]
    z = z[keep]
    w = w[keep]
    w ./= mean(w)

    N = length(z)           # number of sample points
    n = min(div(N, 2)-1, n)   # max order of rational function (number of support points will be n+1)

    reltol = tol*norm(f, Inf)
    SF = Diagonal(f)           # Left scaling matrix

    f, z, w = promote(f, z, w)
    P = promote_type(S, T, U)

    J = collect(1:N)           # Index for non-support points (approximation points)
    zs = P[]                   # Support points
    fs = P[]                   # Function values at support points
    C = P[]

    wtmeanf = sum(f.*w)/sum(w)
    DiagW = Diagonal(w)
    R = f .- wtmeanf
    wvec = nothing
    @inbounds for m = 1:n+1
        j = argmax(abs.((f .- R).*w))               # select next support point
        push!(zs, z[j])
        push!(fs, f[j])
        deleteat!(J, findfirst(isequal(j), J))   # update index vector to remove one new support point

        # Add a column of Cauchy matrix
        if isempty(C)
            C = reshape((1 ./ (z .- z[j])), (N,1))
        else
            C = [C (1 ./ (z .- z[j]))]
        end

        Sf = Diagonal(copy(fs))       # Right scaling matrix
        A = DiagW*(SF*C - C*Sf)       # Loewner matrix
        G = svd(A[J, :])
        wvec = G.V[:, end]               # weight vector = vector with min singular value

        num = C*(wvec.*fs)
        den = C*wvec
        R .= f
        R[J] .= num[J]./den[J]                  # rational approximation

        err = norm((f - R).*w, Inf)
        verbose && println("Iteration: ", m, "  err: ", err)
        err <= reltol && break                # stop if converged
    end
    BarycentricRational(zs, fs, wvec)
end
