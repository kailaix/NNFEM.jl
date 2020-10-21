using Documenter
using NNFEM
using Markdown
makedocs(sitename="NNFEM", modules=[NNFEM],
pages = Any[
    "index.md",
    "Examples: Inverse Problems"=>["verify.md", "verify_linear.md", "hyperelasticity.md", "verify_function.md"],
    "Examples: Forward Computation"=>["ex_simulation.md", "ex_ad.md"],
    "Manual"=>["representation.md", "solvers.md", "gmsh.md"],
    "api.md"
],
authors = "Kailai Xu and Daniel (Zhengyu) Huang")

deploydocs(
    repo = "github.com/kailaix/NNFEM.jl.git",
)