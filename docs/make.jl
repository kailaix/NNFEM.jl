using Documenter
using NNFEM
makedocs(sitename="NNFEM", modules=[NNFEM],
pages = Any[
    "index.md",
    "api.md",
    "plasticity.md"
],
authors = "Kailai Xu and Daniel (Zhengyu) Huang")

deploydocs(
    repo = "github.com/kailaix/NNFEM.jl.git",
)