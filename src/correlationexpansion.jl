module correlationexpansion

import Base: trace, ==, +, -, *, /
import ..operators: dagger, identityoperator,
                    trace, ptrace, normalize!, tensor, permutesystems,
                    gemv!, gemm!

using Combinatorics, Iterators
using ..bases
# using ..states
using ..operators
using ..operators_dense
using ..operators_lazy
using ..ode_dopri

# import Base: *, full
# import ..operators

typealias Mask{N} NTuple{N, Bool}

indices2mask(N::Int, indices::Vector{Int}) = Mask(tuple([(i in indices) for i=1:N]...))
mask2indices{N}(mask::Mask{N}) = Int[i for i=1:N if mask[i]]

complement(N::Int, indices::Vector{Int}) = Int[i for i=1:N if i ∉ indices]
complement{N}(mask::Mask{N}) = tuple([! x for x in mask]...)

correlationindices(N::Int, order::Int) = Set(combinations(1:N, order))
function correlationmasks(N::Int)
    S = Set{Mask{N}}()
    for n=2:N
        S = S ∪ correlationmasks(N, n)
    end
    S
end
function correlationmasks(N::Int, order::Int)
    @assert N > 1
    @assert order > 0
    @assert N >= order
    Set(indices2mask(N, indices) for indices in correlationindices(N, order))
end
correlationmasks{N}(S::Set{Mask{N}}, order::Int) = Set(s for s in S if sum(s)==order)
subcorrelationmasks{N}(mask::Mask{N}) = Set(indices2mask(N, indices) for indices in
        chain([combinations(mask2indices(mask), k) for k=2:sum(mask)-1]...))


"""
An operator including only certain correlations.

operators
    A tuple containing the reduced density matrix of each subsystem.
correlations
    A (mask->operator) dict. A mask is a tuple containing booleans which
    indicate if the corresponding subsystem is included in the correlation.
    The operator is the correlation between the specified subsystems.
"""
type ApproximateOperator{N} <: Operator
    basis_l::CompositeBasis
    basis_r::CompositeBasis
    factor::Complex128
    operators::NTuple{N, DenseOperator}
    correlations::Dict{Mask{N}, DenseOperator}

    function ApproximateOperator{N}(operators::NTuple{N, DenseOperator},
                correlations::Dict{Mask{N}, DenseOperator},
                factor::Number=1)
        basis_l = tensor([op.basis_l for op in operators]...)
        basis_r = tensor([op.basis_r for op in operators]...)
        for (mask, op) in correlations
            @assert sum(mask) > 1
            @assert op.basis_l == tensor(basis_l.bases[[mask...]]...)
            @assert op.basis_r == tensor(basis_r.bases[[mask...]]...)
        end
        new(basis_l, basis_r, factor, operators, correlations)
    end
end

function ApproximateOperator{N}(basis_l::CompositeBasis, basis_r::CompositeBasis, S::Set{Mask{N}})
    operators = ([DenseOperator(basis_l.bases[i], basis_r.bases[i]) for i=1:N]...)
    correlations = Dict{Mask{N}, DenseOperator}()
    for mask in S
        @assert sum(mask) > 1
        correlations[mask] = tensor(operators[[mask...]]...)
    end
    ApproximateOperator{N}(operators, correlations)
end
ApproximateOperator{N}(basis::CompositeBasis, S::Set{Mask{N}}) = ApproximateOperator(basis, basis, S)
function ApproximateOperator{N}(operators::Vector, S::Set{Mask{N}})
    @assert length(operators) == N
    correlations = Dict{Mask{N}, DenseOperator}()
    for op in operators
        @assert typeof(op) <: Operator
    end
    for mask in S
        @assert sum(mask) > 1
        b_l = CompositeBasis([op.basis_l for op in operators[[mask...]]]...)
        b_r = CompositeBasis([op.basis_r for op in operators[[mask...]]]...)
        correlations[mask] = DenseOperator(b_l, b_r)
    end
    ApproximateOperator{N}((operators...), correlations)
end
function ApproximateOperator(operators::Vector)
    N = length(operators)
    ApproximateOperator{N}((operators...), Dict{Mask{N}, DenseOperator}())
end

function datalength{N}(x::ApproximateOperator{N})
    L = sum(Int[length(op.basis_l)*length(op.basis_r) for op in x.operators])
    L += sum(Int[length(op.basis_l)*length(op.basis_r) for op in values(x.correlations)])
    L
end

"""
Tensor product of a correlation and the density operators of the other subsystems.

Arguments
---------
operators
    Tuple containing the reduced density operators of each subsystem.
mask
    A tuple containing booleans specifying if the n-th subsystem is included
    in the correlation.
correlation
    Correlation operator for the subsystems specified by the given mask.
"""
function embedcorrelation{N}(operators::NTuple{N, DenseOperator}, mask::Mask{N},
            correlation::DenseOperator)
    # Product density operator of all subsystems not included in the correlation.
    if sum(mask) == N
        return correlation
    end
    ρ = tensor(operators[[complement(mask)...]]...)
    op = correlation ⊗ ρ # Subsystems are now in wrong order
    perm = sortperm([mask2indices(mask); mask2indices(complement(mask))])
    permutesystems(op, perm)
end
function embedcorrelation{N}(operators::NTuple{N, DenseOperator}, mask::Mask{N},
            correlation::Number)
    @assert sum(mask) == 0
    tensor(operators...)*correlation
end
embedcorrelation{N}(operators::NTuple{N, DenseOperator}, indices::Vector{Int}, correlation) = embedcorrelation(operators, indices2mask(N, indices), correlation)

"""
Calculate the normalized correlation of the subsystems specified by the given index mask.

Arguments
---------
rho
    Density operator of the total system.
mask
    A tuple containing booleans specifying if the n-th subsystem is included
    in the correlation.

Optional Arguments
------------------
operators
    A tuple containing the reduced density operators of the single subsystems.
subcorrelations
    A (mask->operator) dictionary storing already calculated correlations.
"""
function correlation{N}(rho::DenseOperator, mask::Mask{N};
            operators::NTuple{N, DenseOperator}=([ptrace(normalize(rho), complement(N, [i]))
                                                  for i in 1:N]...),
            subcorrelations::Dict{Mask{N}, DenseOperator}=Dict())
    # Check if this correlation was already calculated.
    if mask in keys(subcorrelations)
        return subcorrelations[mask]
    end
    order = sum(mask)
    rho = normalize(rho)
    σ = ptrace(rho, mask2indices(complement(mask)))
    σ -= tensor(operators[[mask...]]...)
    for submask in subcorrelationmasks(mask)
        subcorrelation = correlation(rho, submask;
                                     operators=operators,
                                     subcorrelations=subcorrelations)
        σ -= embedcorrelation((operators[[mask...]]...), submask[[mask...]], subcorrelation)
    end
    subcorrelations[mask] = σ
    σ
end

correlation{N}(rho::ApproximateOperator{N}, mask::Mask{N}) = rho.correlations[mask]


"""
Approximate a density operator by including only certain correlations.

Arguments
---------
rho
    The density operator that should be approximated.
masks
    A set containing an index mask for every correlation that should be
    included. A index mask is a tuple consisting of booleans which indicate
    if the n-th subsystem is included in the correlation.
"""
function approximate{N}(rho::DenseOperator, masks::Set{Mask{N}})
    alpha = trace(rho)
    rho = normalize(rho)
    operators = ([ptrace(rho, complement(N, [i])) for i in 1:N]...)
    subcorrelations = Dict{Mask{N}, DenseOperator}() # Dictionary to store intermediate results
    correlations = Dict{Mask{N}, DenseOperator}()
    for mask in masks
        correlations[mask] = correlation(rho, mask;
                                         operators=operators,
                                         subcorrelations=subcorrelations)
    end
    ApproximateOperator{N}(operators, correlations, alpha)
end
function approximate(rho::DenseOperator)
    @assert typeof(rho.basis_l) == CompositeBasis
    N = length(rho.basis_l.bases)
    approximate(rho, Set{Mask{N}}())
end

ptrace{N}(mask::Mask{N}, indices::Vector{Int}) = (mask[complement(N, indices)]...)

function ptrace{N}(rho::ApproximateOperator{N}, indices::Vector{Int})
    operators = (rho.operators[complement(N, indices)]...)
    factors = [trace(op) for op in rho.operators]
    result = tensor(operators...)*prod(factors[indices])
    for mask in keys(rho.correlations)
        I = mask2indices(mask)
        factor = prod(factors[setdiff(indices, I)])
        if isempty(I ∩ indices)
            correlation = factor*rho.correlations[mask]
        else
            J = [i-sum(complement(N, I).<i) for i in I ∩ indices]
            correlation = factor*ptrace(rho.correlations[mask], J)
        end
        op = embedcorrelation(operators, ptrace(mask, indices), correlation)
        result += op
    end
    rho.factor*result
end

function Base.full{N}(rho::ApproximateOperator{N})
    result = tensor(rho.operators...)
    for (mask, correlation) in rho.correlations
        result += embedcorrelation(rho.operators, mask, correlation)
    end
    rho.factor*result
end

function removeindices{T}(D::Dict{Int,T}, indices::Vector{Int})
    result = Dict{Int,T}()
    for (i, op_i) in D
        @assert i ∉ indices
        j = i-sum(indices.<i)
        result[j] = op_i
    end
    result
end

function *{N}(rho::ApproximateOperator{N}, op::LazyTensor)
    operators = ([i ∈ keys(op.operators) ? rho.operators[i]*op.operators[i] : copy(rho.operators[i]) for i=1:N]...)
    correlations = Dict{Mask{N}, DenseOperator}()
    for mask in keys(rho.correlations)
        I = mask2indices(mask)
        D = Dict(i=>op.operators[i] for i in I ∩ keys(op.operators))
        if isempty(D)
            correlations[mask] = rho.correlations[mask]
        else
            I_ = complement(N, I)
            op_I = embed(ptrace(op.basis_l, I_), ptrace(op.basis_r, I_), removeindices(D, I_))
            correlations[mask] = rho.correlations[mask]*op_I
        end
    end
    ApproximateOperator{N}(operators, correlations, rho.factor*op.factor)
end

function *{N}(op::LazyTensor, rho::ApproximateOperator{N})
    operators = ([i ∈ keys(op.operators) ? op.operators[i]*rho.operators[i] : copy(rho.operators[i]) for i=1:N]...)
    correlations = Dict{Mask{N}, DenseOperator}()
    for mask in keys(rho.correlations)
        I = mask2indices(mask)
        D = Dict(i=>op.operators[i] for i in I ∩ keys(op.operators))
        if isempty(D)
            correlations[mask] = rho.correlations[mask]
        else
            I_ = complement(N, I)
            op_I = embed(ptrace(op.basis_l, I_), ptrace(op.basis_r, I_), removeindices(D, I_))
            correlations[mask] = op_I*rho.correlations[mask]
        end
    end
    ApproximateOperator{N}(operators, correlations, rho.factor*op.factor)
end

*{N}(a::Number, b::ApproximateOperator{N}) = ApproximateOperator{N}(b.operators, b.correlations, a*b.factor)
*{N}(a::ApproximateOperator{N}, b::Number) = ApproximateOperator{N}(a.operators, a.correlations, a.factor*b)

function issupermask{N}(mask::Mask{N}, supermask::Mask{N})
    for i=1:N
        if mask[i] && !supermask[i]
            return false
        end
    end
    true
end

function embedcorrelation{T<:Operator}(operators::Dict{Vector{Int}, T})
    indices, operator_list = zip(operators...)
    indices = collect(indices)
    perm = sortperm(indices, by=x->x[1])
    indices = [indices[perm]...;]
    operator_list = collect(operator_list)[perm]
    # println(typeof(operator_list))
    # println(indices)
    # @assert Set(indices) == Set([1:length(indices);])
    op = tensor(operator_list...)
    if indices == [1:length(indices);]
        # println("No permute")
        return op
    else
        # println("permute")
        perm = sortperm(indices)
        return permutesystems(op, perm)
    end
end

function _ptrace{N}(rho::ApproximateOperator{N}, H::LazySum,
                    H_rho::Vector{ApproximateOperator{N}}, rho_H::Vector{ApproximateOperator{N}},
                    Gamma::Matrix{Float64}, J::Vector{LazyTensor}, Jdagger::Vector{LazyTensor},
                    indices::Vector{Int})
    I_trace_compl = complement(N, indices)
    basis_l = ptrace(rho.basis_l, indices)
    basis_r = ptrace(rho.basis_r, indices)
    result = DenseOperator(basis_l, basis_r)
    h_index = 0
    for (a, h) in zip(H.factors, H.operators)
        h_index += 1
        I_op = collect(keys(h.operators))
        # println(I_op)
        J = setdiff(I_op, indices)
        if length(J)==0
            continue
        end
        # println(typeof(i=>h.operators[i]*rho.operators[i] for i in J))
        # println(J)
        h_rho = Dict{Int,DenseOperator}(i=>H_rho[h_index].operators[i] for i in J)
        rho_h = Dict{Int,DenseOperator}(i=>rho_H[h_index].operators[i] for i in J)
        factor = -1im*h.factor*a
        factors = Dict{Int,Complex128}(i=>trace(H_rho[h_index].operators[i]) for i in I_op ∩ indices)
        f = factor*prod(values(factors))
        result += f*tensor([i ∈ J ? h_rho[i] : rho.operators[i] for i=1:N if i ∉ indices]...)
        result -= f*tensor([i ∈ J ? rho_h[i] : rho.operators[i] for i=1:N if i ∉ indices]...)
        for (mask, correlation) in rho.correlations
            I_cor = mask2indices(mask)
            if length(setdiff(indices ∩ I_cor, I_op)) != 0
                continue
            end
            # h_ = tensor([i ∈ I_op ? h.operators[i] : identityoperator(DenseOperator, h.basis_l.bases[i], h.basis_r.bases[i]) for i in I_cor]...)
            I_cor_ = [i for i=1:length(I_trace_compl) if I_trace_compl[i] ∈ I_cor]
            # h_σ = Dict(I_cor_=>ptrace(h_*correlation, [i for i=1:length(I_cor) if I_cor[i] in indices]))
            # σ_h = Dict(I_cor_=>ptrace(correlation*h_, [i for i=1:length(I_cor) if I_cor[i] in indices]))
            h_σ = Dict(I_cor_=>ptrace(H_rho[h_index].correlations[mask], [i for i=1:length(I_cor) if I_cor[i] in indices]))
            σ_h = Dict(I_cor_=>ptrace(rho_H[h_index].correlations[mask], [i for i=1:length(I_cor) if I_cor[i] in indices]))
            for i in I_trace_compl
                if i ∈ I_cor
                    continue
                elseif i ∈ I_op
                    j = [i-sum(indices.<i)]
                    h_σ[j] = h_rho[i]
                    σ_h[j] = rho_h[i]
                else
                    j = [i-sum(indices.<i)]
                    h_σ[j] = rho.operators[i]
                    σ_h[j] = rho.operators[i]
                end
            end
            f = factor*prod(Complex128[factors[i] for i in setdiff(I_op ∩ indices, I_cor)])
            result += f*embedcorrelation(h_σ)
            result -= f*embedcorrelation(σ_h)
        end
    end
    # A = LazySum([-1im*a*(h*rho) for (a, h) in zip(H.factors, H.operators)]...)
    # A -= LazySum([-1im*a*(rho*h) for (a, h) in zip(H.factors, H.operators)]...)
    # for j=1:length(J), i=1:length(J)
    #     A += Gamma[i,j]*lazy(J[i]*rho*Jdagger[j])
    #     A -= Gamma[i,j]*0.5*lazy(Jdagger[j]*(J[i]*rho))
    #     A -= Gamma[i,j]*0.5*lazy((rho*Jdagger[j])*J[i])
    # end
    result
end

function ptrace_recursive{N}(rho::ApproximateOperator{N}, H::LazySum,
                    h_rho::Vector{ApproximateOperator{N}}, rho_h::Vector{ApproximateOperator{N}},
                    Gamma::Matrix{Float64}, J::Vector{LazyTensor}, Jdagger::Vector{LazyTensor},
                    mask::Mask{N}, P::Dict{Mask{N}, DenseOperator})
    supermasks = [s for s in keys(P) if issupermask(mask, s)]
    if length(supermasks)==0
        result = _ptrace(rho, H, h_rho, rho_h, Gamma, J, Jdagger, mask2indices(complement(mask)))
    else
        i = indmin([length(P[s].basis_l)*length(P[s].basis_r) for s in supermasks])
        supermask = supermasks[i]
        indices = mask2indices(complement(mask[[supermask...]]))
        result = ptrace(P[supermask], indices)
    end
    result
end

function ptraces{N}(rho::ApproximateOperator{N}, H::LazySum,
                    Gamma::Matrix{Float64}, J::Vector{LazyTensor}, Jdagger::Vector{LazyTensor})
    correlations = Dict{Mask{N}, DenseOperator}()
    h_rho = ApproximateOperator{N}[h*rho for h in H.operators]
    rho_h = ApproximateOperator{N}[rho*h for h in H.operators]
    for n=N:-1:2
        for mask in [s for s in keys(rho.correlations) if sum(s)==n]
            correlations[mask] = ptrace_recursive(rho, H, h_rho, rho_h, Gamma, J, Jdagger, mask, correlations)
        end
    end
    operators = ([ptrace_recursive(rho, H, h_rho, rho_h, Gamma, J, Jdagger, indices2mask(N, [i]), correlations) for i=1:N]...)
    operators, correlations
end

function dmaster{N}(rho::ApproximateOperator{N}, H::LazySum,
                    Gamma::Matrix{Float64}, J::Vector{LazyTensor}, Jdagger::Vector{LazyTensor})
    # A = LazySum([-1im*a*(h*rho) for (a, h) in zip(H.factors, H.operators)]...)
    # A -= LazySum([-1im*a*(rho*h) for (a, h) in zip(H.factors, H.operators)]...)
    # for j=1:length(J), i=1:length(J)
    #     A += Gamma[i,j]*lazy(J[i]*rho*Jdagger[j])
    #     A -= Gamma[i,j]*0.5*lazy(Jdagger[j]*(J[i]*rho))
    #     A -= Gamma[i,j]*0.5*lazy((rho*Jdagger[j])*J[i])
    # end
    doperators, dcorrelations = ptraces(rho, H, Gamma, J, Jdagger)
    for order=2:N
        for mask in correlationmasks(Set(keys(rho.correlations)), order)
            I = mask2indices(mask)
            suboperators = rho.operators[I]
            # Tr{̇̇d/dt ρ}
            σ_I = dcorrelations[mask]
            # d/dt ρ^{s_k}
            for i = 1:order
                σ_I -= embedcorrelation(suboperators, [i], doperators[I[i]])
            end
            # d/dt σ^{s}
            for submask in keys(dcorrelations) ∩ subcorrelationmasks(mask)
                σ_I -= embedcorrelation(suboperators, (submask[I]...), dcorrelations[submask])
                for i in setdiff(mask2indices(complement(submask)), mask2indices(complement(mask)))
                    ops = ([i==j ? doperators[j] : rho.operators[j] for j in I]...)
                    σ_I -= embedcorrelation(ops, (submask[I]...), rho.correlations[submask])
                end
            end
            dcorrelations[mask] = σ_I
        end
    end
    return ApproximateOperator{N}(doperators, dcorrelations, rho.factor)
end

function as_vector{N}(rho::ApproximateOperator{N}, x::Vector{Complex128})
    @assert length(x) == datalength(rho)
    i = 0
    for op in rho.operators
        L_i = length(op.basis_l)*length(op.basis_r)
        x[i+1:i+L_i] = reshape(op.data, L_i)
        i += L_i
    end
    for mask in sort(collect(keys(rho.correlations)))
        op = rho.correlations[mask]
        L_i = length(op.basis_l)*length(op.basis_r)
        x[i+1:i+L_i] = reshape(op.data, L_i)
        i += L_i
    end
    x
end

function as_operator{N}(x::Vector{Complex128}, rho::ApproximateOperator{N})
    @assert length(x) == datalength(rho)
    i = 0
    for op in rho.operators
        L_i = length(op.basis_l)*length(op.basis_r)
        reshape(op.data, L_i)[:] = x[i+1:i+L_i]
        i += L_i
    end
    for mask in sort(collect(keys(rho.correlations)))
        op = rho.correlations[mask]
        L_i = length(op.basis_l)*length(op.basis_r)
        reshape(op.data, L_i)[:] = x[i+1:i+L_i]
        i += L_i
    end
    rho
end

function integrate_master{N}(dmaster::Function, tspan, rho0::ApproximateOperator{N};
                fout::Union{Function,Void}=nothing, kwargs...)
    x0 = as_vector(rho0, zeros(Complex128, datalength(rho0)))
    f = (x->x)
    if fout==nothing
        tout = Float64[]
        xout = ApproximateOperator{N}[]
        function fout_(t, rho::ApproximateOperator{N})
            push!(tout, t)
            push!(xout, deepcopy(rho))
        end
        f = fout_
    else
        f = fout
    end
    tmp = deepcopy(rho0)
    f_(t, x::Vector{Complex128}) = f(t, as_operator(x, tmp))
    ode(dmaster, float(tspan), x0, f_; kwargs...)
    return fout==nothing ? (tout, xout) : nothing
end

function master{N}(tspan, rho0::ApproximateOperator{N}, H::LazySum, J::Vector{LazyTensor};
                    Gamma::Union{Vector{Float64}, Matrix{Float64}}=eye(Float64, length(J), length(J)),
                fout::Union{Function,Void}=nothing,
                kwargs...)
    Jdagger = LazyTensor[dagger(j) for j=J]
    rho = deepcopy(rho0)
    function dmaster_(t, x::Vector{Complex128}, dx::Vector{Complex128})
        drho = dmaster(as_operator(x, rho), H, Gamma, J, Jdagger)
        as_vector(drho, dx)
    end
    integrate_master(dmaster_, tspan, rho0; fout=fout, kwargs...)
end

master(tspan, rho0, H::LazyTensor, J; kwargs...) = master(tspan, rho0, LazySum(H), J; kwargs...)



end # module