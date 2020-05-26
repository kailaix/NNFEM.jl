#-------------------------------------------------
#  generate MMS using the following code
# using SymPy
# x, y = @vars x y
# u = x*y*(1-y)
# v = x*(1-y)

# ux = diff(u, x)
# uy = diff(u, y)
# vx = diff(v, x)
# vy = diff(v, y)

# ε = [
#     ux
#     vy
#     uy + vx 
# ]

# H = domain.elements[1].mat[1].H
# σ = H*ε

# f1 = diff(σ[1], x) + diff(σ[3], y)
# f2 = diff(σ[3], x) + diff(σ[2], y)


using NNFEM
using PoreFlow
using PyPlot 

m = 40
n = 40
h = 1/m
domain = example_domain(m, n, h)
x, y = domain.nodes[:,1], domain.nodes[:,2]
out = [@. x*y*(1-y)
      @. x*(1-y)]


EBC = zeros(Int64,domain.nnodes, 2)
g = [(@. x*y*(1-y)) (@. x*(1-y))]
FBC = zeros(Int64,domain.nnodes,2)
f = zeros(domain.nnodes,2)

for j = 1:n+1
    EBC[(j-1)*(m+1)+m+1, :] .= -1
    EBC[(j-1)*(m+1)+1, :] .= -1
end

for i = 1:m+1
    EBC[n*(m+1)+i, :] .= -1
    EBC[i, :] .= -1
end

nodes, elements = domain.nodes, domain.elements

domain = Domain(nodes, elements, 2, EBC, g, FBC, f)
H = domain.elements[1].mat[1].H

EBC_func = nothing 
FBC_func = nothing 

function body_func(x, y, t)
    f1 = @. -1.48148148148148x - 2.46913580246914
    f2 = @. 2.46913580246914 - 4.93827160493827y
    -[f1 f2]
end
globaldata = GlobalData(missing, missing, missing, missing, domain.neqs, EBC_func, FBC_func, body_func)

#-------------------------------------------------
# PoreFlow Solver
@info "Solving using PoreFlow..."
bd = bcnode("all", m, n, h)
K = compute_fem_stiffness_matrix(H, m, n, h)
K, Kbd = fem_impose_Dirichlet_boundary_condition(K, bd, m, n, h)
ff = Kbd * domain.state[[bd; bd.+domain.nnodes]]

F1 = eval_f_on_gauss_pts((x,y)->-1.48148148148148x - 2.46913580246914, m, n, h)
F2 = eval_f_on_gauss_pts((x,y)-> 2.46913580246914 - 4.93827160493827y, m, n, h)
F = compute_fem_source_term(-F1, -F2, m, n, h)

rhs = F - ff
rhs[[bd; bd.+domain.nnodes]] = out[[bd; bd.+domain.nnodes]]
out2 = K\rhs
figure(figsize=(10,10))
title("PoreFlow")
subplot(221)
visualize_scalar_on_scoped_body(out2[1:domain.nnodes], zeros(size(domain.state)...), domain)
subplot(222)
visualize_scalar_on_scoped_body(out[1:domain.nnodes], zeros(size(domain.state)...), domain)

subplot(223)
visualize_scalar_on_scoped_body(out2[domain.nnodes+1:end], zeros(size(domain.state)...), domain)
subplot(224)
visualize_scalar_on_scoped_body(out[domain.nnodes+1:end], zeros(size(domain.state)...), domain)

#-------------------------------------------------
# NNFEM Solver
@info "Solving using NNFEM..."
globaldata, domain = LinearStaticSolver(globaldata, domain)
figure(figsize=(10,10))
title("PoreFlow")
subplot(221)
visualize_scalar_on_scoped_body(domain.state[1:domain.nnodes], zeros(size(domain.state)...), domain)
subplot(222)
visualize_scalar_on_scoped_body(out[1:domain.nnodes], zeros(size(domain.state)...), domain)
subplot(223)
visualize_scalar_on_scoped_body(domain.state[domain.nnodes+1:end], zeros(size(domain.state)...), domain)
subplot(224)
visualize_scalar_on_scoped_body(out[domain.nnodes+1:end], zeros(size(domain.state)...), domain)
