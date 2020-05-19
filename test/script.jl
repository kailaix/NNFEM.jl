using Revise
using NNFEM
using PoreFlow
using LinearAlgebra
using PyPlot



f = eval_f_on_gauss_pts((x,y)->x^2+y^2+1, m, n, h)
fext = compute_body_force_terms1(domain, (x,y)->x^2+y^2+1)
pf_f = compute_fem_source_term1(f, m, n, h)


