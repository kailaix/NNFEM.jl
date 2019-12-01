# Instructions for Running Benchmarks

## `Plate/MustiScale2`

1. Generate Data with `GenerateData.sh`. 

2. Pretrain NN with Linear Models
```bash
julia NNPreLSfit.jl
```

3. Check `NNPreLSfit.jl` results
```bash
julia Adj_Loss.jl
```

4. 