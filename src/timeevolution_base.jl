# using ..ode_dopri
using OrdinaryDiffEq, DiffEqBase

function recast! end

"""
df(t, state::T, dstate::T)
"""
function integrate{T}(tspan::Vector{Float64}, df::Function, x0::Vector{Complex128},
            state::T, dstate::T, fout::Function; kwargs...)
    function df_(t, x::Vector{Complex128}, dx::Vector{Complex128})
        recast!(x, state)
        recast!(dx, dstate)
        df(t, state, dstate)
        recast!(dstate, dx)
    end
    function fout_(integrator)
        t = integrator.t
        x = integrator.u
        recast!(x, state)
        fout(t, state)
    end
    cb = DiscreteCallback((t, u, integrator)->true, fout_; save_positions=(false, false))
    prob = ODEProblem(df_, x0, (tspan[1], tspan[end]))
    sol = solve(prob, Tsit5(), callback=cb, saveat=tspan, dense=false, abstol=1e-8, reltol=1e-6)
    # sol = solve(prob, Tsit5(), kwargs=kwargs...)
    # Save only on points in tspan by calling fout_ ??
    # (sol.t, sol.u) ??
end

function integrate{T}(tspan::Vector{Float64}, df::Function, x0::Vector{Complex128},
            state::T, dstate::T, ::Void; kwargs...)
    tout = Float64[]
    xout = T[]
    function fout(t::Float64, state::T)
        push!(tout, t)
        push!(xout, copy(state))
    end
    integrate(tspan, df, x0, state, dstate, fout; kwargs...)
    (tout, xout)
end