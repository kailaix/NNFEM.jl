
# # using SymPy

# x,y,t = @vars x y t
# u = exp(-t)*x*(1-x)*y*(1-y)
# v = u 
# ux = diff(u, x)
# uy = diff(u, y)
# vx = diff(v, x)
# vy = diff(v, y)
# e = -[ux    
#     vy
#     uy+vx
#     ]
# σ = Cs * e
# f1 = -(diff(σ[1], x) + diff(σ[3], y)) + u
# print(replace(replace(sympy.julia_code(simplify(f1)), ".^"=>"^"), ".*"=>"*"))

# f2 = -(diff(σ[3], x) + diff(σ[2], y)) + u
# print(replace(replace(sympy.julia_code(simplify(f2)), ".^"=>"^"), ".*"=>"*"))


@testset "viscous" begin 
    domain = example_domain(10, 10, 0.1)
    
    EBC = zeros(Int64, domain.nnodes, 2)
    FBC = zeros(Int64, domain.nnodes, 2)
    g = zeros(domain.nnodes, 2)
    f = zeros(domain.nnodes, 2)
    for i = 1:domain.nnodes
        if domain.nodes[i,1]<1e-5 || domain.nodes[i,2]<1e-5 || domain.nodes[i,1]>1-1e-5 || domain.nodes[i,2]>1-1e-5
            EBC[i,:] .= -1
        end
    end

    Cs = [3. 2. 0.0
        2. 5. 0.0
        0.0 0.0 1.0]
    Hs = zeros(3, 3)


    domain = Domain(domain.nodes, domain.elements, 2, EBC, g, FBC, f)
    globdat = example_global_data(domain)
    globdat.Body_func = (x, y, t)->begin
        f1 = @. 1.0*(1.0*x^2*y^2 - 1.0*x^2*y + 2.0*x^2 - 1.0*x*y^2 + 13.0*x*y - 8.0*x + 6.0*y^2 - 12.0*y + 3.0)*exp(-t)
        f2 = @. 1.0*(1.0*x^2*y^2 - 1.0*x^2*y + 10.0*x^2 - 1.0*x*y^2 + 13.0*x*y - 16.0*x + 2.0*y^2 - 8.0*y + 3.0)*exp(-t)
        out = [f1 f2]
    end
    Δt = 0.01
    NT = 100

    ts = GeneralizedAlphaSolverTime(Δt, NT)
    ubd, abd = compute_boundary_info(domain, globdat, ts)
    Fext = compute_external_force(domain, globdat, ts)
    
    # xy = reshape(domain.nodes[domain.dof_to_eq], sum(domain.dof_to_eq)÷2, 2)
    x, y = domain.nodes[:,1], domain.nodes[:,2]
    d0 = @. x*(1-x)*y*(1-y)
    v0 = - @. x*(1-x)*y*(1-y)
    a0 = @. x*(1-x)*y*(1-y)
    d0 = [d0;d0]
    v0 = [v0;v0]
    a0 = [a0;a0]
    
    
    d, v, a = GeneralizedAlphaSolver(globdat, domain, d0, v0, a0, Δt, NT, Cs, Hs, Fext, ubd, abd)

    sess = Session(); init(sess)
    d_ = run(sess, d)

    uexact = zeros(NT+1, domain.nnodes*2)
    for i = 1:NT+1
        t = (i-1)*Δt
        uexact[i,:] = [(@. exp(-t) * x*(1-x)*y*(1-y));(@. exp(-t) * x*(1-x)*y*(1-y))]
    end

    uexact - d_
    

end 