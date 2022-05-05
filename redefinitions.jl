using TimerOutputs
to = TimerOutput()

import NLsolve.nlsolve
@timeit to "nlsolve" function NLsolve.nlsolve(f,
        initial_x::AbstractArray;
        method::Symbol = :trust_region,
        autodiff = :central,
        inplace = !applicable(f, initial_x),
        kwargs...)
    if method in (:anderson, :broyden)
        df = NLsolve.NonDifferentiable(f, initial_x, copy(initial_x); inplace=inplace)
    else
        df = NLsolve.OnceDifferentiable(f, initial_x, copy(initial_x); autodiff=autodiff, inplace=inplace)
    end

    NLsolve.nlsolve(df, initial_x; method = method, kwargs...)
end

import DiffEqBase.__solve
function DiffEqBase.__solve(prob::BVProblem, alg::Shooting; kwargs...)
    bc = prob.bc
    u0 = deepcopy(prob.u0)
    # Form a root finding function.
    loss = function (resid, minimizer)
        uEltype = eltype(minimizer)
        tmp_prob = remake(prob,u0=minimizer)
        @timeit to "forward_solve" sol = solve(tmp_prob,alg.ode_alg;kwargs...)
        bc(resid,sol,sol.prob.p,sol.t)
        nothing
    end
    opt = alg.nlsolve(loss, u0)
    sol_prob = remake(prob,u0=opt[1])
    sol = solve(sol_prob, alg.ode_alg;kwargs...)
    if sol.retcode == opt[2]
        DiffEqBase.solution_new_retcode(sol,:Success)
    else
        DiffEqBase.solution_new_retcode(sol,:Failure)
    end
    sol
end

import Base.:/
function (Base.:/)(to::TimerOutput, n::Real)
    tof = TimerOutputs.flatten(copy(to))

    for inner_timer in values(tof.inner_timers)
        data = inner_timer.accumulated_data
        # TimerOutputs.jl only supports Ints for these, so get two places
        #  of precision (need to divide out afterwards)
        data.ncalls = cld(data.ncalls * 100, n)
        data.time = cld(data.time * 100, n)
        data.allocs = cld(data.time * 100, n)
    end
    tof
end
