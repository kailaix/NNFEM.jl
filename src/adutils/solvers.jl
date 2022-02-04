export ExplicitSolver, ExplicitSolverTime, GeneralizedAlphaSolver, 
GeneralizedAlphaSolverTime, ExplicitStaticSolver, ImplicitStaticSolver,
compute_boundary_info, compute_external_force


@doc raw"""
    ExplicitSolverTime(Δt::Float64, NT::Int64)

Returns the times for explicit solver. Boundary conditions and external forces should be given at these times.
"""
function ExplicitSolverTime(Δt::Float64, NT::Int64)
    U = zeros(NT)
    for i = 1:NT 
        U[i] = i*Δt
    end
    U 
end

@doc raw"""
    ExplicitSolver(globdat::GlobalData, domain::Domain,
        d0::Union{Array{Float64, 1}, PyObject}, 
        v0::Union{Array{Float64, 1}, PyObject}, 
        a0::Union{Array{Float64, 1}, PyObject}, 
        Δt::Float64, NT::Int64, 
        H::Union{Array{Float64, 3}, Array{Float64, 2}, PyObject},
        Fext::Union{Array{Float64, 2}, PyObject, Missing}=missing,
        ubd::Union{Array{Float64, 2}, PyObject, Missing}=missing,
        abd::Union{Array{Float64, 2}, PyObject, Missing}=missing; strain::String = "small")

Differentiable Explicit Solver. 

- `d0`, `v0`, `a0`: initial **full** displacement, velocity, and acceleration. 

- `Δt`: time step 

- `Hs`: linear elasticity matrix at each Gauss point 

- `Fext`: external force, $\mathrm{NT}\times n$, where $n$ is the active dof.
   The external force includes all **body forces**, **external load forces** (also called **edge forces** in NNFEM) and **boundary acceleration-induced forces**.

- `ubd`, `abd`: boundary displacementt and acceleration, $\mathrm{NT}\times m$, where $m$ is **time-dependent** boundary DOF. 
   Time-independent boundary conditions are extracted from `domain`. 

- `strain_type` (default = "small"): small strain or finite strain
"""
function ExplicitSolver(globdat::GlobalData, domain::Domain,
    d0::Union{Array{Float64, 1}, PyObject}, 
    v0::Union{Array{Float64, 1}, PyObject}, 
    a0::Union{Array{Float64, 1}, PyObject}, 
    Δt::Float64, NT::Int64, 
    H::Union{Array{Float64, 3}, Array{Float64, 2}, PyObject},
    Fext::Union{Array{Float64, 2}, PyObject, Missing}=missing,
    ubd::Union{Array{Float64, 2}, PyObject, Missing}=missing,
    abd::Union{Array{Float64, 2}, PyObject, Missing}=missing; 
    strain_type::String = "small")
    assembleMassMatrix!(globdat, domain)
    if !(strain_type in ["small", "finite"])
        error("Only small strain or finite strain are supported. Unknown strain type $strain_type")
    end
    init_nnfem(domain)
    M = factorize(constant(globdat.M))
    bddof = findall(domain.EBC[:] .== -2)
    fixed_bddof = findall(domain.EBC[:] .== -1)

    Fext, ubd, abd, H = convert_to_tensor([Fext, ubd, abd, H], [Float64, Float64, Float64, Float64])

    function condition(i, tas...)
        i<=NT
    end
    function body(i, tas...)
        d_arr, v_arr, a_arr = tas
        u, ∂u, ∂∂u = read(d_arr, i), read(v_arr, i), read(a_arr, i)

        u +=  Δt*∂u + 0.5*Δt*Δt*∂∂u
        ∂u += 0.5*Δt * ∂∂u

        if !ismissing(abd)
            u = scatter_update(u, bddof, ubd[i])
        end
        if length(sum(fixed_bddof))>0
            u = scatter_update(u, fixed_bddof, d0[fixed_bddof])
        end

        if strain_type=="small"
            ε = s_eval_strain_on_gauss_points(u, domain)
        else
            ε = f_eval_strain_on_gauss_points(u, domain)
        end
        if length(size(H))==2
            σ = tf.matmul(ε, H)
        else
            σ = batch_matmul(H, ε)
        end 
        if strain_type=="small"
            fint  = s_compute_internal_force_term(σ, domain)
        else 
            fint  = f_compute_internal_force_term(σ, u, domain)
        end
        if ismissing(Fext)
            fext = zeros(length(fint))
        else
            fext = Fext[i]
        end
        ∂∂up = vector(findall(domain.dof_to_eq), M\(fext - fint), domain.nnodes*2)
        
        if !ismissing(abd)
            ∂∂up = scatter_update(∂∂up, bddof, abd[i])
        end

        ∂u += 0.5 * Δt * ∂∂up

        i+1, write(d_arr, i+1, u), write(v_arr, i+1, ∂u), write(a_arr, i+1, ∂∂up)
    end

    arr_d = TensorArray(NT+1); arr_d = write(arr_d, 1, d0)
    arr_v = TensorArray(NT+1); arr_v = write(arr_v, 1, v0)
    arr_a = TensorArray(NT+1); arr_a = write(arr_a, 1, a0)
    i = constant(1, dtype=Int32)
    tas = [arr_d, arr_v, arr_a]
    _, d, v, a = while_loop(condition, body, [i, tas...])
    d, v, a = stack(d), stack(v), stack(a)
    sp = (NT+1, 2domain.nnodes)
    set_shape(d, sp), set_shape(v, sp), set_shape(a, sp)
end


"""
    compute_boundary_info(domain::Domain, globdat::GlobalData, ts::Array{Float64})

Computes the boundary information `ubd` and `abd`
"""
function compute_boundary_info(domain::Domain, globdat::GlobalData, ts::Array{Float64})
    bd = domain.EBC[:] .== -2 
    if sum(bd)==0
        return missing, missing 
    end

    @assert globdat.EBC_func!=nothing 

    ubd = zeros(length(ts), sum(bd))
    abd = zeros(length(ts), sum(bd))
    for i = 1:length(ts)
        time = ts[i]
        ubd[i,:], _, abd[i,:] = globdat.EBC_func(time) # user defined time-dependent boundary
    end

    return ubd, abd
end

"""
    compute_external_force(domain::Domain, globdat::GlobalData, ts::Union{Array{Float64}, Missing} = missing)

Computes the external force (body force, edge force and force due to boundary acceleration).
"""
function compute_external_force(domain::Domain, globdat::GlobalData, ts::Union{Array{Float64}, Missing} = missing)
    if ismissing(ts)
        globdat.time = 0.0 
        return getExternalForce(domain, globdat)
    end

    NT = length(ts)
    fext = zeros(NT, domain.neqs)
    for i = 1:length(ts)
        time = ts[i]
        globdat.time = time 
        fext[i, :] = getExternalForce(domain, globdat)
    end
    globdat.time = 0.0
    fext
end
compute_external_force(globdat::GlobalData, domain::Domain, ts::Union{Array{Float64}, Missing} = missing) = compute_external_force(domain, globdat, ts)

@doc raw"""
    GeneralizedAlphaSolverTime(Δt::Float64, NT::Int64;ρ::Float64 = 0.0)

Returns the times for the generalized $\alpha$ solver. Boundary conditions and external forces should be given at these times.
"""
function GeneralizedAlphaSolverTime(Δt::Float64, NT::Int64;ρ::Float64 = 0.0)
    U = zeros(NT)
    @assert 0<=ρ<=1
    αm = (2ρ-1)/(1+ρ)
    αf = ρ/(1+ρ)    
    β2 = 0.5*(1 - αm + αf)^2
    γ = 0.5 - αm + αf
    t = 0
    for i = 1:NT 
        t += (1 - αf)*Δt 
        U[i] = t
    end
    U 
end


@doc raw"""
    GeneralizedAlphaSolver(globdat::GlobalData, domain::Domain,
        d0::Union{Array{Float64, 1}, PyObject}, 
        v0::Union{Array{Float64, 1}, PyObject}, 
        a0::Union{Array{Float64, 1}, PyObject}, 
        Δt::Float64, NT::Int64, 
        Hs::Union{Array{Float64, 3}, Array{Float64, 2}, PyObject},
        Fext::Union{Array{Float64, 2}, PyObject, Missing}=missing,
        ubd::Union{Array{Float64, 2}, PyObject, Missing}=missing,
        abd::Union{Array{Float64, 2}, PyObject, Missing}=missing; ρ::Float64 = 0.0)

Differentiable Generalized $\alpha$ scheme. This is an extension of [`αscheme`](https://kailaix.github.io/ADCME.jl/dev/alphascheme/)
provided in ADCME. This function does not support damping and variable time step (for efficiency). 

- `d0`, `v0`, `a0`: initial **full** displacement, velocity, and acceleration. 

- `Δt`: time step 

- `Hs`: linear elasticity matrix at each Gauss point 

- `Fext`: external force, $\mathrm{NT}\times n$, where $n$ is the active dof. 
  The external force includes all **body forces**, **external load forces** (also called **edge forces** in NNFEM) and **boundary acceleration-induced forces**.

- `ubd`, `abd`: boundary displacementt and acceleration, $\mathrm{NT}\times m$, where $m$ is boundary DOF. 
  Time-independent boundary conditions are extracted from `domain`. 

`GeneralizedAlphaSolver` does not support finite-strain continuum yet.  
"""
function GeneralizedAlphaSolver(globdat::GlobalData, domain::Domain,
    d0::Union{Array{Float64, 1}, PyObject}, 
    v0::Union{Array{Float64, 1}, PyObject}, 
    a0::Union{Array{Float64, 1}, PyObject}, 
    Δt::Float64, NT::Int64, 
    Hs::Union{Array{Float64, 3}, Array{Float64, 2}, PyObject},
    Fext::Union{Array{Float64, 2}, PyObject, Missing}=missing,
    ubd::Union{Array{Float64, 2}, PyObject, Missing}=missing,
    abd::Union{Array{Float64, 2}, PyObject, Missing}=missing; ρ::Float64 = 0.0)
    assembleMassMatrix!(globdat, domain)
    @assert 0<=ρ<=1
    init_nnfem(domain)
    αm = (2ρ-1)/(1+ρ)
    αf = ρ/(1+ρ)    
    β2 = 0.5*(1 - αm + αf)^2
    γ = 0.5 - αm + αf

    Fext, ubd, abd, Hs = convert_to_tensor([Fext, ubd, abd, Hs], [Float64, Float64, Float64, Float64])
    M = constant(globdat.M)
    stiff = s_compute_stiffness_matrix(Hs, domain)
    A = M*(1 - αm) + (1 - αf) * 0.5 * β2 * Δt^2 * stiff
    A = factorize(A)
    bddof = findall(domain.EBC[:] .== -2)
    fixed_bddof = findall(domain.EBC[:] .== -1)
    nbddof = findall(domain.dof_to_eq)

    function condition(i, tas...)
        i<=NT
    end
    function body(i, tas...)
        d_arr, v_arr, a_arr = tas 
        u, ∂u, ∂∂u = read(d_arr, i), read(v_arr, i), read(a_arr, i)
        if ismissing(Fext)
            fext = zeros(domain.neqs)
        else
            fext = Fext[i]
        end
        ∂∂up = ∂∂u
        up =  (1 - αf)*(u + Δt*∂u + 0.5 * Δt * Δt * ((1 - β2)*∂∂u + β2*∂∂up)) + αf*u
        if !ismissing(abd)
            up = scatter_update(up, bddof, ubd[i])
        end
        if length(sum(fixed_bddof))>0
            up = scatter_update(up, fixed_bddof, d0[fixed_bddof])
        end

        ε = s_eval_strain_on_gauss_points(up, domain)
        if length(size(Hs))==2
            σ = tf.matmul(ε, Hs)
        else
            # @info Hs, ε
            σ = batch_matmul(Hs, ε)
        end 
        fint  = s_compute_internal_force_term(σ, domain)
        res = M * (∂∂up[nbddof] *(1 - αm) + αm*∂∂u[nbddof])  + fint - fext
        Δ = -(A\res)
        ∂∂up= scatter_add(∂∂up, nbddof, Δ)
        if !ismissing(abd)
            ∂∂up = scatter_update(∂∂up, bddof, abd[i])
        end

        # updaet 
        u += Δt * ∂u + Δt^2/2 * ((1 - β2) * ∂∂u + β2 * ∂∂up)
        ∂u += Δt * ((1 - γ) * ∂∂u + γ * ∂∂up)

        if length(sum(fixed_bddof))>0
            u = scatter_update(u, fixed_bddof, d0[fixed_bddof])
        end

        i+1, write(d_arr, i+1, u), write(v_arr, i+1, ∂u), write(a_arr, i+1, ∂∂up)
    end

    arr_d = TensorArray(NT+1); arr_d = write(arr_d, 1, d0)
    arr_v = TensorArray(NT+1); arr_v = write(arr_v, 1, v0)
    arr_a = TensorArray(NT+1); arr_a = write(arr_a, 1, a0)
    i = constant(1, dtype=Int32)
    tas = [arr_d, arr_v, arr_a]
    _, d, v, a = while_loop(condition, body, [i, tas...])
    d, v, a = stack(d), stack(v), stack(a)
    sp = (NT+1, 2domain.nnodes)
    set_shape(d, sp), set_shape(v, sp), set_shape(a, sp)
end


@doc raw"""
    GeneralizedAlphaSolver(globdat::GlobalData, domain::Domain,
        d0::Union{Array{Float64, 1}, PyObject}, 
        v0::Union{Array{Float64, 1}, PyObject}, 
        a0::Union{Array{Float64, 1}, PyObject}, 
        Δt::Float64, NT::Int64, 
        Cs::Union{Array{Float64, 3}, Array{Float64, 2}, PyObject},
        Hs::Union{Array{Float64, 3}, Array{Float64, 2}, PyObject},
        Fext::Union{Array{Float64, 2}, PyObject, Missing}=missing,
        ubd::Union{Array{Float64, 2}, PyObject, Missing}=missing,
        abd::Union{Array{Float64, 2}, PyObject, Missing}=missing; ρ::Float64 = 0.0)

Solve linear dynamical structural problem using a generalized alpha solver. 
`Cs` and `Hs` are the corresponding linear viscosity and linear elasticity matrix. 
"""
function GeneralizedAlphaSolver(globdat::GlobalData, domain::Domain,
    d0::Union{Array{Float64, 1}, PyObject}, 
    v0::Union{Array{Float64, 1}, PyObject}, 
    a0::Union{Array{Float64, 1}, PyObject}, 
    Δt::Float64, NT::Int64, 
    Cs::Union{Array{Float64, 3}, Array{Float64, 2}, PyObject},
    Hs::Union{Array{Float64, 3}, Array{Float64, 2}, PyObject},
    Fext::Union{Array{Float64, 2}, PyObject, Missing}=missing,
    ubd::Union{Array{Float64, 2}, PyObject, Missing}=missing,
    abd::Union{Array{Float64, 2}, PyObject, Missing}=missing; ρ::Float64 = 0.0)
    assembleMassMatrix!(globdat, domain)
    @assert 0<=ρ<=1
    init_nnfem(domain)
    αm = (2ρ-1)/(1+ρ)
    αf = ρ/(1+ρ)    
    β2 = 0.5*(1 - αm + αf)^2
    γ = 0.5 - αm + αf

    Cs, Fext, ubd, abd, Hs = convert_to_tensor([Cs, Fext, ubd, abd, Hs], [Float64, Float64, Float64, Float64, Float64])
    M = constant(globdat.M)
    if length(size(Hs))==2
        Hs = reshape(repeat(Hs, getNGauss(domain)), (getNGauss(domain), 3, 3))
    end
    if length(size(Cs))==2
        Cs = reshape(repeat(Cs, getNGauss(domain)), (getNGauss(domain), 3, 3))
    end
    
    stiff = s_compute_stiffness_matrix(Hs, domain)
    damp = s_compute_stiffness_matrix(Cs, domain)
    A = M*(1 - αm) + (1-αf)*Δt*γ*damp +  (1 - αf) * 0.5 * β2 * Δt^2 * stiff
    A = factorize(A)
    bddof = findall(domain.EBC[:] .== -2)
    fixed_bddof = findall(domain.EBC[:] .== -1)
    nbddof = findall(domain.dof_to_eq)

    function condition(i, tas...)
        i<=NT
    end
    function body(i, tas...)
        d_arr, v_arr, a_arr = tas 
        u, ∂u, ∂∂u = read(d_arr, i), read(v_arr, i), read(a_arr, i)
        if ismissing(Fext)
            fext = zeros(domain.neqs)
        else
            fext = Fext[i]
        end
        ∂∂up = ∂∂u
        up =  (1 - αf)*(u + Δt*∂u + 0.5 * Δt * Δt * ((1 - β2)*∂∂u + β2*∂∂up)) + αf*u
        if !ismissing(abd)
            up = scatter_update(up, bddof, ubd[i])
        end
        if length(sum(fixed_bddof))>0
            up = scatter_update(up, fixed_bddof, d0[fixed_bddof])
        end

        ε = s_eval_strain_on_gauss_points(up, domain)
        if length(size(Hs))==2
            σ = tf.matmul(ε, Hs)
        else
            # @info Hs, ε
            σ = batch_matmul(Hs, ε)
        end 
        fint  = s_compute_internal_force_term(σ, domain)

        vn = ∂u + Δt * (1- γ)*∂∂u
        vf = (1-αf)*vn + αf*∂u
        dotε = s_eval_strain_on_gauss_points(vf, domain)
        if length(size(Cs))==2
            visc = tf.matmul(dotε, Cs)
        else
            visc = batch_matmul(Cs, dotε)
        end
        fv = s_compute_internal_force_term(visc, domain)

        
        res = M * (∂∂up[nbddof] *(1 - αm) + αm*∂∂u[nbddof])  + fv + fint - fext
        Δ = -(A\res)
        ∂∂up= scatter_add(∂∂up, nbddof, Δ)
        if !ismissing(abd)
            ∂∂up = scatter_update(∂∂up, bddof, abd[i])
        end

        # updaet 
        u += Δt * ∂u + Δt^2/2 * ((1 - β2) * ∂∂u + β2 * ∂∂up)
        ∂u += Δt * ((1 - γ) * ∂∂u + γ * ∂∂up)

        if length(sum(fixed_bddof))>0
            u = scatter_update(u, fixed_bddof, d0[fixed_bddof])
        end

        i+1, write(d_arr, i+1, u), write(v_arr, i+1, ∂u), write(a_arr, i+1, ∂∂up)
    end

    arr_d = TensorArray(NT+1); arr_d = write(arr_d, 1, d0)
    arr_v = TensorArray(NT+1); arr_v = write(arr_v, 1, v0)
    arr_a = TensorArray(NT+1); arr_a = write(arr_a, 1, a0)
    i = constant(1, dtype=Int32)
    tas = [arr_d, arr_v, arr_a]
    _, d, v, a = while_loop(condition, body, [i, tas...])
    d, v, a = stack(d), stack(v), stack(a)
    sp = (NT+1, 2domain.nnodes)
    set_shape(d, sp), set_shape(v, sp), set_shape(a, sp)
end



############################# NN based constitutive models #############################


@doc raw"""
    ExplicitSolver(globdat::GlobalData, domain::Domain,
        d0::Union{Array{Float64, 1}, PyObject}, 
        v0::Union{Array{Float64, 1}, PyObject}, 
        a0::Union{Array{Float64, 1}, PyObject}, 
        Δt::Float64, NT::Int64, 
        nn::Function,
        Fext::Union{Array{Float64, 2}, PyObject, Missing}=missing,
        ubd::Union{Array{Float64, 2}, PyObject, Missing}=missing,
        abd::Union{Array{Float64, 2}, PyObject, Missing}=missing; strain_type::String = "small"))

Similar to [`ExplicitSolver`](@ref); however, the constituve relation from $\epsilon$ to $\sigma$ must be provided by 
the function `nn`.
"""
function ExplicitSolver(globdat::GlobalData, domain::Domain,
    d0::Union{Array{Float64, 1}, PyObject}, 
    v0::Union{Array{Float64, 1}, PyObject}, 
    a0::Union{Array{Float64, 1}, PyObject}, 
    Δt::Float64, NT::Int64, 
    nn::Function,
    Fext::Union{Array{Float64, 2}, PyObject, Missing}=missing,
    ubd::Union{Array{Float64, 2}, PyObject, Missing}=missing,
    abd::Union{Array{Float64, 2}, PyObject, Missing}=missing; strain_type::String = "small")
    assembleMassMatrix!(globdat, domain)
    if !(strain_type in ["small", "finite"])
        error("Only small strain or finite strain are supported. Unknown strain type $strain_type")
    end

    init_nnfem(domain)
    M = factorize(constant(globdat.M))
    bddof = findall(domain.EBC[:] .== -2)
    fixed_bddof = findall(domain.EBC[:] .== -1)

    Fext, ubd, abd = convert_to_tensor([Fext, ubd, abd], [Float64, Float64, Float64])

    function condition(i, tas...)
        i<=NT
    end
    function body(i, tas...)
        d_arr, v_arr, a_arr = tas
        u, ∂u, ∂∂u = read(d_arr, i), read(v_arr, i), read(a_arr, i)

        u +=  Δt*∂u + 0.5*Δt*Δt*∂∂u
        ∂u += 0.5*Δt * ∂∂u

        if !ismissing(abd)
            u = scatter_update(u, bddof, ubd[i])
        end

        if length(sum(fixed_bddof))>0
            u = scatter_update(u, fixed_bddof, d0[fixed_bddof])
        end

        if strain_type=="small"
            ε = s_eval_strain_on_gauss_points(u, domain)
        else
            ε = f_eval_strain_on_gauss_points(u, domain)
        end
        
        σ = nn(ε)

        if strain_type=="small"
            fint  = s_compute_internal_force_term(σ, domain)
        else 
            fint  = f_compute_internal_force_term(σ, u, domain)
        end
        if ismissing(Fext)
            fext = zeros(length(fint))
        else
            fext = Fext[i]
        end
        ∂∂up = vector(findall(domain.dof_to_eq), M\(fext - fint), domain.nnodes*2)
        
        if !ismissing(abd)
            ∂∂up = scatter_update(∂∂up, bddof, abd[i])
        end
        if length(sum(fixed_bddof))>0
            ∂∂up = scatter_update(∂∂up, fixed_bddof, a0[fixed_bddof])
        end


        ∂u += 0.5 * Δt * ∂∂up

        i+1, write(d_arr, i+1, u), write(v_arr, i+1, ∂u), write(a_arr, i+1, ∂∂up)
    end

    arr_d = TensorArray(NT+1); arr_d = write(arr_d, 1, d0)
    arr_v = TensorArray(NT+1); arr_v = write(arr_v, 1, v0)
    arr_a = TensorArray(NT+1); arr_a = write(arr_a, 1, a0)
    i = constant(1, dtype=Int32)
    tas = [arr_d, arr_v, arr_a]
    _, d, v, a = while_loop(condition, body, [i, tas...])
    d, v, a = stack(d), stack(v), stack(a)
    sp = (NT+1, 2domain.nnodes)
    set_shape(d, sp), set_shape(v, sp), set_shape(a, sp)
end

@doc raw"""
    ExplicitSolver(globdat::GlobalData, domain::Domain,
        d0::Union{Array{Float64, 1}, PyObject}, 
        v0::Union{Array{Float64, 1}, PyObject}, 
        a0::Union{Array{Float64, 1}, PyObject}, 
        σ0::Union{Array{Float64, 1}, PyObject}, 
        ε0::Union{Array{Float64, 1}, PyObject}, 
        Δt::Float64, NT::Int64, 
        nn::Function,
        Fext::Union{Array{Float64, 2}, PyObject, Missing}=missing,
        ubd::Union{Array{Float64, 2}, PyObject, Missing}=missing,
        abd::Union{Array{Float64, 2}, PyObject, Missing}=missing; strain_type::String = "small")

Similar to [`ExplicitSolver`](@ref); however, the constitutive relation has the form 

$$\sigma^{n+1} = \mathrm{nn}(\epsilon^{n+1}, \epsilon^n, \sigma^n)$$

Here the strain and stress are $n \times 3$ tensors. $n$ is the total number of Gaussian points and can be 
obtained via `getNGauss(domain)`.
"""
function ExplicitSolver(globdat::GlobalData, domain::Domain,
    d0::Union{Array{Float64, 1}, PyObject}, 
    v0::Union{Array{Float64, 1}, PyObject}, 
    a0::Union{Array{Float64, 1}, PyObject}, 
    σ0::Union{Array{Float64, 2}, PyObject}, 
    ε0::Union{Array{Float64, 2}, PyObject}, 
    Δt::Float64, NT::Int64, 
    nn::Function,
    Fext::Union{Array{Float64, 2}, PyObject, Missing}=missing,
    ubd::Union{Array{Float64, 2}, PyObject, Missing}=missing,
    abd::Union{Array{Float64, 2}, PyObject, Missing}=missing; strain_type::String = "small")
    assembleMassMatrix!(globdat, domain)
    if !(strain_type in ["small", "finite"])
        error("Only small strain or finite strain are supported. Unknown strain type $strain_type")
    end

    init_nnfem(domain)
    M = factorize(constant(globdat.M))
    bddof = findall(domain.EBC[:] .== -2)
    fixed_bddof = findall(domain.EBC[:] .== -1)

    Fext, ubd, abd = convert_to_tensor([Fext, ubd, abd], [Float64, Float64, Float64])

    function condition(i, tas...)
        i<=NT
    end
    function body(i, tas...)
        d_arr, v_arr, a_arr, σ_arr, ε_arr = tas
        u, ∂u, ∂∂u = read(d_arr, i), read(v_arr, i), read(a_arr, i)
        σc, εc = read(σ_arr, i), read(ε_arr, i)

        u +=  Δt*∂u + 0.5*Δt*Δt*∂∂u
        ∂u += 0.5*Δt * ∂∂u

        if !ismissing(abd)
            u = scatter_update(u, bddof, ubd[i])
        end
        if length(sum(fixed_bddof))>0
            u = scatter_update(u, fixed_bddof, d0[fixed_bddof])
        end

        if strain_type=="small"
            ε = s_eval_strain_on_gauss_points(u, domain)
        else
            ε = f_eval_strain_on_gauss_points(u, domain)
        end

        σ = nn(ε, εc, σc)
        if strain_type=="small"
            fint  = s_compute_internal_force_term(σ, domain)
        else 
            fint  = f_compute_internal_force_term(σ, u, domain)
        end

        if ismissing(Fext)
            fext = zeros(length(fint))
        else
            fext = Fext[i]
        end
        ∂∂up = vector(findall(domain.dof_to_eq), M\(fext - fint), domain.nnodes*2)
        
        if !ismissing(abd)
            ∂∂up = scatter_update(∂∂up, bddof, abd[i])
        end
        if length(sum(fixed_bddof))>0
            ∂∂up = scatter_update(∂∂up, fixed_bddof, a0[fixed_bddof])
        end

        ∂u += 0.5 * Δt * ∂∂up

        i+1, write(d_arr, i+1, u), write(v_arr, i+1, ∂u), write(a_arr, i+1, ∂∂up), write(σ_arr, i+1, σ), write(ε_arr, i+1, ε)
    end

    arr_d = TensorArray(NT+1); arr_d = write(arr_d, 1, d0)
    arr_v = TensorArray(NT+1); arr_v = write(arr_v, 1, v0)
    arr_a = TensorArray(NT+1); arr_a = write(arr_a, 1, a0)
    arr_σ = TensorArray(NT+1); arr_σ = write(arr_σ, 1, σ0)
    arr_ε = TensorArray(NT+1); arr_ε = write(arr_ε, 1, ε0)
    i = constant(1, dtype=Int32)
    tas = [arr_d, arr_v, arr_a, arr_σ, arr_ε]
    _, d, v, a, σ, ε = while_loop(condition, body, [i, tas...])
    d, v, a, σ, ε = stack(d), stack(v), stack(a), stack(σ), stack(ε)
    sp = (NT+1, 2domain.nnodes)
    set_shape(d, sp), set_shape(v, sp), set_shape(a, sp), set_shape(σ, (NT+1, getNGauss(domain),3)), set_shape(ε, (NT+1, getNGauss(domain),3))
end


function ExplicitStaticSolver(globdat::GlobalData, domain::Domain,
    d0::Union{Array{Float64, 1}, PyObject}, 
    v0::Union{Array{Float64, 1}, PyObject}, 
    Δt::Float64, NT::Int64, 
    nn::Function,
    Fext::Union{Array{Float64, 1}, PyObject, Missing}=missing; strain_type::String = "small")
    assembleMassMatrix!(globdat, domain)
    @assert length(findall(domain.EBC[:] .== -2))==0
    @assert length(findall(domain.FBC[:] .== -2))==0
    a0 = tf.zeros_like(d0)
    Fext = repeat(constant(reshape(Fext, 1, :)), NT, 1)
    load_vec = reshape(Array((1:NT)/NT), NT, 1)
    Fext = Fext .* load_vec
    d, _, _ = ExplicitSolver(globdat, domain, d0, v0, a0, Δt, NT, nn, Fext, missing, missing; strain_type = strain_type)
    return d
end

@doc raw"""
    LinearStaticSolver(globdat::GlobalData, domain::Domain,
        d0::Union{Array{Float64, 1}, PyObject}, 
        Hs::Union{Array{Float64, 3}, Array{Float64, 2}, PyObject},
        Fext::Union{Array{Float64, 1}, PyObject, Missing}=missing)

An AD-capable static linear elasticity solver. 

- `d0` has length `2domain.nnodes` (full DOFs)

- `Hs` is either a $3\times 3$ matrix or $N\times 3\times 3$ tensor, representing elasticity matrix. 
"""
function LinearStaticSolver(globdat::GlobalData, domain::Domain,
    d0::Union{Array{Float64, 1}, PyObject}, 
    Hs::Union{Array{Float64, 3}, Array{Float64, 2}, PyObject},
    Fext::Union{Array{Float64, 1}, PyObject, Missing}=missing)
    @assert length(findall(domain.EBC[:] .== -2))==0
    @assert length(findall(domain.FBC[:] .== -2))==0
    init_nnfem(domain)
    Hs = constant(Hs)
    if length(size(Hs))==2
        Hs = repeat(reshape(Hs, (-1,)), getNGauss(domain))
        Hs = reshape(Hs, (getNGauss(domain), 3, 3) )
    end
    Fext = coalesce(Fext, zeros(domain.neqs))
    d0, Fext = convert_to_tensor([d0, Fext], [Float64, Float64])
    ε = s_eval_strain_on_gauss_points(d0, domain)
    σ = batch_matmul(Hs, ε)
    fint = s_compute_internal_force_term(σ, domain)
    stiff = s_compute_stiffness_matrix(Hs, domain)
    u0 = stiff\(Fext - fint)
    nbdnode = findall(domain.EBC[:].!=-1)
    d = scatter_update(d0, nbdnode, u0)
    return d
end

function ImplicitStaticSolver(globdat::GlobalData, domain::Domain,
    d0::Union{Array{Float64, 1}, PyObject}, 
    Δt::Float64, NT::Int64, 
    Hs::Union{Array{Float64, 3}, Array{Float64, 2}, PyObject},
    Fext::Union{Array{Float64, 1}, PyObject, Missing}=missing; 
    strain_type::String = "small", method::String = "newton")
    if !(strain_type in ["small", "finite"])
        error("Only small strain or finite strain are supported. Unknown strain type $strain_type")
    end
    assembleMassMatrix!(globdat, domain)
    @assert length(findall(domain.EBC[:] .== -2))==0
    @assert length(findall(domain.FBC[:] .== -2))==0
    
    Hs = constant(Hs)
    if length(size(Hs))==2
        Hs = repeat(reshape(Hs, (-1,)), getNGauss(domain))
        Hs = reshape(Hs, (getNGauss(domain), 3, 3) )
    end
    if method=="incremental"
        a0 = tf.zeros_like(d0)
        v0 = tf.zeros_like(d0)
        Fext = repeat(constant(reshape(Fext, 1, :)), NT, 1)
        d, _, _ = GeneralizedAlphaSolver(globdat, domain, d0, v0, a0, Δt, NT, Hs, Fext, missing, missing)
    elseif method=="newton"
        if !(strain_type in ["small"])
            error("Only small strain or finite strain are supported. Unknown strain type $strain_type")
        end
        d0, Fext = convert_to_tensor([d0, Fext], [Float64, Float64])
        stiff = s_compute_stiffness_matrix(Hs, domain)
        u0 = stiff\Fext
        nbdnode = findall(domain.EBC[:].!=-1)
        d = scatter_add(d0, nbdnode, u0)
    end
    return d
end


@doc raw"""
    ImplicitStaticSolver(globdat::GlobalData, domain::Domain,
        d0::Union{Array{Float64, 1}, PyObject}, 
        nn::Function, θ::Union{Array{Float64, 1}, PyObject},
        Fext::Union{Array{Float64, 1}, PyObject, Missing}=missing)


Solves the static problem 

$$K(u) = F$$

using Newton's method. Users provide `nn`, which is a function that outputs stress and stress sensitivity given the strain tensor. 

$$\sigma, \frac{\partial \sigma}{\partial \epsilon} = \mathrm{nn}(\epsilon, \theta)$$

- `d0`: a vector of length `2domain.nnodes`, the fixed Dirichlet DOF should be populated with boundary values. 

- `Fext`: external force, a vector of length `domain.neqs`
"""
function ImplicitStaticSolver(globdat::GlobalData, domain::Domain,
    d0::Union{Array{Float64, 1}, PyObject}, 
    nn::Function, θ::Union{Array{Float64, 1}, PyObject},
    Fext::Union{Array{Float64, 1}, PyObject, Missing}=missing)
    init_nnfem(domain)
    assembleMassMatrix!(globdat, domain)
    @assert length(findall(domain.EBC[:] .== -2))==0
    @assert length(findall(domain.FBC[:] .== -2))==0
    nbdnode = findall(domain.EBC[:].!=-1)
    K0 = spdiag(domain.nnodes*2)
    function residual_and_jac(θ, u)
        ε = s_eval_strain_on_gauss_points(u, domain)
        σ, Hs = nn(ε, θ)
        stiff = s_compute_stiffness_matrix(Hs, domain)
        fint = s_compute_internal_force_term(σ, domain)
        # @info fint, σ, Hs, stiff
        δ = scatter_update(d0, nbdnode, fint - Fext)
        K = scatter_update(K0, nbdnode, nbdnode, stiff)
        # @info δ, K  
        δ, K  
    end
    d0, Fext, θ = convert_to_tensor([d0, Fext, θ], [Float64, Float64, Float64])
    u0 = newton_raphson_with_grad(residual_and_jac, d0, θ)
    return u0
end