cd(@__DIR__)
using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()
using Documenter
using NNFEM
makedocs(sitename="NNFEM", modules=[NNFEM],
pages = Any[
    "index.md",
    "Examples"=>["ex_simulation.md"]
    "api.md",
],
authors = "Kailai Xu and Daniel (Zhengyu) Huang")

deploydocs(
    repo = "github.com/kailaix/NNFEM.jl.git",
)