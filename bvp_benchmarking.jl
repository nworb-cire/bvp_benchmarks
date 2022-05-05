using Flux, DiffEqFlux, DifferentialEquations, BoundaryValueDiffEq, Zygote, TimerOutputs
using BenchmarkTools
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 30

using LoggingExtras
logger = FileLogger("logs/$adjoint_name-$HIDDEN_DIM.log")
global_logger(logger)

const NODE_ORDER = 4
include("adjoints.jl")
SENSEALG = adjoints[adjoint_name]

nn = FastChain(
	FastDense(NODE_ORDER+1, HIDDEN_DIM, relu),
	FastDense(HIDDEN_DIM, 1)
)
θ₀ = initial_params(nn)
@info "$(length(θ₀)) parameters"

dudt(u,p,t) = [u[2:end] ; nn([u;t], p)]
function boundary!(residual,u,p,t)
	residual[1] = u[1][1]
	residual[2:end] = u[end][2:end]
end

include("gt.jl")

function loss(θ)
	pred = solve(
		TwoPointBVProblem(
			dudt,
			boundary!,
			randn(NODE_ORDER),
			(0., 1.),
			θ
		),
		saveat = ts,
		sensealg = SENSEALG
	)
	ŷ = pred[1,:]
	return Flux.mae(ŷ, gt)
end

function loss(θ, u₀)
	pred = solve(
		ODEProblem(
			dudt,
			u₀,
			(0., 1.),
			θ
		),
		saveat = ts,
		sensealg = SENSEALG
	)
	ŷ = pred[1,:]
	return Flux.mae(ŷ, gt)
end

include("redefinitions.jl")

function show_trial(t::BenchmarkTools.Trial)::String
	io = IOBuffer()
	show(io, MIME"text/plain"(), t)
	s = String(take!(io))
	close(io)
	return s
end

try
	reset_timer!(to)
	trial = @benchmark loss($θ₀)
	N = length(trial.times)
	@info "Forward" samples=N trial=show_trial(trial) avg=(to / N)
	reset_timer!(to)
catch
	@info "Forward failed."
end

try
	reset_timer!(to)
	trial = @benchmark gradient(loss, $θ₀)
	N = length(trial.times)
	@info "Gradient" samples=N trial=show_trial(trial) avg=(to / N)
	reset_timer!(to)
catch
	@info "Gradient failed."
end

try
	reset_timer!(to)
	trial = @benchmark let
		u₀ = solve(BVProblem(dudt, boundary!, randn(NODE_ORDER), (0., 1.), θ₀))[1]
		gradient($θ₀) do θ
			loss(θ, u₀)
		end
	end
	N = length(trial.times)
	@info "Gradient bypass" samples=N trial=show_trial(trial) avg=(to / N)
	reset_timer!(to)
catch
	@info "Gradient bypass failed."
end
