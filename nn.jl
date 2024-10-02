using Lux, Random, Optimisers, Zygote, BenchmarkTools, Functors

to_bigfloat = p -> Functors.fmap(x -> map(BigFloat, x), p)

rng = Random.default_rng()
Random.seed!(rng, 0)
model = Chain(
    Dense(128, 256, tanh),
    Dense(256, 1, tanh),
    Dense(1, 10)
)
ps, st = Lux.setup(rng, model)
ps = to_bigfloat(ps)

x = rand(rng, BigFloat, 128, 2)
y, _ = Lux.apply(model, x, ps, st)
y[3,1]