# Object BarycentricRational represents a rational function in barycentric form.
#
# The form is f(z) = (Σ i=1:p w_i f_i/(z-λ[i]))/(Σ i=1:p w_i/(λ-λ[i]))
# where λ[1:p] are the p support points
# f[1:p] are the function values at the supports
# w[1:p] are the weights
#
# See _Numerical Recipes, 3rd Edition_ (2007) or similar.
#

using LinearAlgebra

"""
    struct BarycentricRational

Represent an order (p,p) rational function in barycentric form. That is, as

f(z) = (Σ i=1:p w[i]*f[i]/(z-λ[i]))/(Σ i=1:p w[i]/(z-λ[i]))
- λ[1:p] are the p support points
- f[1:p] are the function values at the supports
- w[1:p] are the weights

# Constructor

    BarycentricRational(λ::AbstractArray{S}, f::AbstractArray{T}, w::AbstractArray{U})

The lengths of `λ`, `f`, and `w` must be equal. Element types will be promoted to be the same.
Poles and roots will be computed automatically and stored as attributes `poles` and `roots`.
"""
struct BarycentricRational{T,U <: AbstractArray}
    λ::T  # support points
    f::T  # value at support points
    w::T  # weights

    poles::U  # poles of the function
    roots::U  # roots of the function
end

# Constructor
function BarycentricRational(λ::AbstractArray{S}, f::AbstractArray{T}, w::AbstractArray{U}) where {S, T, U}
    λ, f, w = promote(λ, f, w)
    P = promote_type(S, T, U)

    n = length(λ)
    if n != length(f)
        throw(DimensionMismatch("length(λ) $(length(λ)) != length(f) $(length(f))"))
    elseif n != length(w)
        throw(DimensionMismatch("length(λ) $(length(λ)) != length(w) $(length(w))"))
    end

    poles = roots_pfrac0(w, λ)
    roots = roots_pfrac0(w.*f, λ)
    if eltype(poles) != eltype(roots)
        poles = complex(poles)
        roots = complex(roots)
    end
    BarycentricRational(λ, f, w, poles, roots)
end

(br::BarycentricRational)(zz) = brat_eval(zz, br)

"""
    brat_eval(z, B::BarycentricRational)

Evaluate the barycetric rational function `B` at `z`, which may be a number or an AbstractArray.
Returns as number or array of the same form and shape as `z` (but promoted to floating point).
"""
function brat_eval(z, br::BarycentricRational)
    arg_isscalar = size(z) == ()
    zvec = arg_isscalar ? [z] : vec(z)
    C = 1.0 ./ (zvec .- transpose(br.λ))   # Cauchy matrix
    f = (C * (br.w .* br.f)) ./ (C * br.w)
    f[isinf.(zvec)] .= sum(br.f .* br.w) / sum(br.w)

    ii = findall(isnan.(f))   # find any values NaN = Inf/Inf, indicating z is at a support point
    for j in ii
        if !isnan(zvec[j]) && ((v = findfirst(isequal(zvec[j]), br.λ)) !== nothing)
            f[j] = br.f[v]  # replace removable singularity at a support point.
        end
    end
    f = arg_isscalar ? f[1] : reshape(f, size(z))
end


"""
    roots_pfrac0(w::AbstractVector, x::AbstractVector)

Returns roots of a partial fraction that sums to zero, specifically
the `n-1` solutions `p` to the equation:

`0 = Σ i=1:n w[i]/(p-x[i])`

There are `n-1` solutions, where `n` is the length of the vectors of weights `w` and of nodes `x`.
"""
function roots_pfrac0(w::AbstractVector, x::AbstractVector)
    n = length(w)
    if n != length(x)
        throw(ErrorException("roots_pfrac0(w, x) has length(w)=$(length(w)) != length(x)=$(length(x))."))
    end
    M = Diagonal(x) - w/sum(w)*x'
    v = eigvals(M)
    # The vector v will contain the (n-1) desired roots, plus 0. Remove the 0.
    _, idx = findmin(abs2.(v))
    deleteat!(v, idx)
    v
end

"""
    roots_pfrac1(w::AbstractVector, x::AbstractVector)

Returns roots of a partial fraction that sums to one, specifically
the solutions `p` to the equation:

`1 = Σ i=1:n w[i]/(p-x[i])`

There are `n` solutions, where `n` is the length of the vectors of weights `w` and of nodes `x`.
"""
function roots_pfrac1(w::AbstractVector, x::AbstractVector)
    n = length(w)
    if n != length(x)
        throw(ErrorException("roots_pfrac1(w, x) has length(w)=$(length(w)) != length(x)=$(length(x))."))
    end
    M = Diagonal(x) + w*ones(n)'
    r = eigvals(M)
    for i=1:n
        # Take up to 3 Newton steps
        for iter = 1:3
            f = sum(w./(r[i].-x))-1
            f1 = -sum(w./(r[i].-x).^2)
            step = f/f1
            r[i] -= step
            abs(step) < 1e-13*max(1, abs(r[i])) && break
        end
    end
    r
end

"""
    roots_pfrac(w::AbstractVector, x::AbstractVector, s::Number)

Returns roots of a partial fraction that sums to `s`, specifically
the solutions `p` to the equation:

`s = Σ i=1:n w[i]/(p-x[i])`

There are `n` solutions, where `n` is the length of the vectors of weights `w` and of nodes `x`.
When `s==0`, however, we omit the implied solution at `p → ∞` and return the other `n-1` roots.
"""
function roots_pfrac(w::AbstractVector, x::AbstractVector, s::Number)
    if s==0
        return roots_pfrac0(w, x)
    end
    roots_pfrac1(w/s, x)
end
