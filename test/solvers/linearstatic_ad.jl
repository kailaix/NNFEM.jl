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
# NNFEM Solver
@info "Solving using AD-free solver..."
globaldata, domain = LinearStaticSolver(globaldata, domain)
figure(figsize=(10,10))
title("NNFEM")
subplot(221)
visualize_scalar_on_scoped_body(domain.state[1:domain.nnodes], zeros(size(domain.state)...), domain)
subplot(222)
visualize_scalar_on_scoped_body(out[1:domain.nnodes], zeros(size(domain.state)...), domain)
subplot(223)
visualize_scalar_on_scoped_body(domain.state[domain.nnodes+1:end], zeros(size(domain.state)...), domain)
subplot(224)
visualize_scalar_on_scoped_body(out[domain.nnodes+1:end], zeros(size(domain.state)...), domain)

@info "Solving using AD-capable solver..."
Fext = compute_external_force(globaldata, domain)
d = LinearStaticSolver(globaldata, domain, domain.state, H, Fext)
sess = Session(); init(sess)
domain.state = run(sess, d)

figure(figsize=(10,10))
subplot(221)
visualize_scalar_on_scoped_body(domain.state[1:domain.nnodes], zeros(size(domain.state)...), domain)
subplot(222)
visualize_scalar_on_scoped_body(out[1:domain.nnodes], zeros(size(domain.state)...), domain)
subplot(223)
visualize_scalar_on_scoped_body(domain.state[domain.nnodes+1:end], zeros(size(domain.state)...), domain)
subplot(224)
visualize_scalar_on_scoped_body(out[domain.nnodes+1:end], zeros(size(domain.state)...), domain)