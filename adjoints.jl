using DiffEqSensitivity
adjoints = Dict{String, Any}(
    "forward"               => ForwardDiffSensitivity(),

    "quad"                  => QuadratureAdjoint(),
    "quad_reverse"          => QuadratureAdjoint(autojacvec = ReverseDiffVJP(true)),
    "quad_zygote"           => QuadratureAdjoint(autojacvec = ZygoteVJP()),
    
    "backsolve"             => BacksolveAdjoint(),
    "backsolve_reverse"     => BacksolveAdjoint(autojacvec = ReverseDiffVJP(true)),
    "backsolve_zygote"      => BacksolveAdjoint(autojacvec = ZygoteVJP()),

    "interp"                => InterpolatingAdjoint(),
    "interp_reverse"        => InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true)),
    "interp_zygote"         => InterpolatingAdjoint(autojacvec = ZygoteVJP()),

    "reverse"               => ReverseDiffAdjoint(),
)
