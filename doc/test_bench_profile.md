# Testing, benchmarks and profiling

Fluidsim comes with command-line tools for testing, benchmarking and profiling. Here are
useful commands:

```bash
fluidsim -h
```

## Testing

```bash
fluidsim-test -h
fluidsim-test -v
```

## Benchmarks

```bash
fluidsim-bench -h
fluidsim-bench -s ns2d
```

```bash
fluidsim-bench-analysis -h
```

## Profiling

```bash
fluidsim-profile -h
fluidsim-profile -s ns2d
fluidsim-profile -p -sf /tmp/fluidsim_profile/result_bench_ns2d_512x512_2017-11-19_22-19-26_8772.json
```
