using Documenter
using NNFEM
makedocs(sitename="NNFEM", modules=[NNFEM],
pages = Any[
    "index.md",
    "api.md",
    "method.md",
    "plasticity.md",
    "instructions.md",
    "api.md"
],
authors = "Kailai Xu and Daniel (Zhengyu) Huang")

deploydocs(
    repo = "github.com/kailaix/NNFEM.jl.git",
)