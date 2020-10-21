export ViscoelasticitySolver

@doc raw"""
    ViscoelasticitySolver(globdat::GlobalData, domain::Domain,
        d0::Union{Array{Float64, 1}, PyObject}, 
        v0::Union{Array{Float64, 1}, PyObject}, 
        a0::Union{Array{Float64, 1}, PyObject}, 
        σ0::Union{Array{Float64, 2}, PyObject}, 
        ε0::Union{Array{Float64, 2}, PyObject}, 
        Δt::Float64, NT::Int64, 
        μ::Union{Array{Float64, 1}, PyObject}, 
        λ::Union{Array{Float64, 1}, PyObject}, 
        η::Union{Array{Float64, 1}, PyObject}, 
        Fext::Union{Array{Float64, 2}, PyObject, Missing}=missing,
        ubd::Union{Array{Float64, 2}, PyObject, Missing}=missing,
        abd::Union{Array{Float64, 2}, PyObject, Missing}=missing; ρ::Float64 = 0.0)

Solves the Maxwell viscoelasticity model using the generalized $\alpha$ scheme.

The constitutive relation has the form 

$$\dot \sigma_{ij} + \frac{\mu}{\eta} \left( \sigma_{ij} - \frac{\sigma_{kk}}{3}\delta_{ij} \right) = 2\mu \dot \epsilon_{ij} + \lambda \dot\epsilon_{kk}\delta_{ij}$$

along with the balance of linear momentum equation

$$\mathrm{div}\ \sigma_{ij,j} + \rho f_i = \rho \ddot u_i$$

Users need to provide $\mu$, $\eta$, $\lambda$ and appropriate boundary/initial conditions. 
"""
function ViscoelasticitySolver(globdat::GlobalData, domain::Domain,
    d0::Union{Array{Float64, 1}, PyObject}, 
    v0::Union{Array{Float64, 1}, PyObject}, 
    a0::Union{Array{Float64, 1}, PyObject}, 
    σ0::Union{Array{Float64, 2}, PyObject}, 
    ε0::Union{Array{Float64, 2}, PyObject}, 
    Δt::Float64, NT::Int64, 
    μ::Union{Array{Float64, 1}, PyObject}, 
    λ::Union{Array{Float64, 1}, PyObject}, 
    η::Union{Array{Float64, 1}, PyObject}, 
    Fext::Union{Array{Float64, 2}, PyObject, Missing}=missing,
    ubd::Union{Array{Float64, 2}, PyObject, Missing}=missing,
    abd::Union{Array{Float64, 2}, PyObject, Missing}=missing; ρ::Float64 = 0.0)
    N = getNGauss(domain)
    @assert length(μ)==length(λ)==length(η)==N 
    S, H = compute_maxwell_viscoelasticity_matrices(μ, λ, η, Δt)
    
    assembleMassMatrix!(globdat, domain)
    @assert 0<=ρ<=1
    init_nnfem(domain)
    αm = (2ρ-1)/(1+ρ)
    αf = ρ/(1+ρ)    
    β2 = 0.5*(1 - αm + αf)^2
    γ = 0.5 - αm + αf

    Fext, ubd, abd = convert_to_tensor([Fext, ubd, abd], [Float64, Float64, Float64])
    M = constant(globdat.M)
    
    stiff = s_compute_stiffness_matrix(H, domain)
    A = M*(1 - αm) +  (1 - αf) * 0.5 * β2 * Δt^2 * stiff
    A = factorize(A)
    bddof = findall(domain.EBC[:] .== -2)
    fixed_bddof = findall(domain.EBC[:] .== -1)
    nbddof = findall(domain.dof_to_eq)

    function condition(i, tas...)
        i<=NT
    end
    function body(i, tas...)
        d_arr, v_arr, a_arr, σ_arr, ε_arr = tas 
        u, ∂u, ∂∂u, σc, εc = read(d_arr, i), read(v_arr, i), read(a_arr, i), read(σ_arr, i), read(ε_arr, i)
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

        # Here ϵc should be interpreted as the true values at (1-αf)*Δt
        ε = s_eval_strain_on_gauss_points(up, domain)
        σ = batch_matmul(H, ε-εc) + batch_matmul(S, σc) 

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

        # We use ϵ to approximate values at (1-αf)*Δt at this time step.
        ε = s_eval_strain_on_gauss_points(u, domain)
        σ = batch_matmul(H, ε-εc) + batch_matmul(S, σc) 

        if length(sum(fixed_bddof))>0
            u = scatter_update(u, fixed_bddof, d0[fixed_bddof])
        end

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
    sp2 = (NT+1, getNGauss(domain), 3)
    set_shape(d, sp), set_shape(v, sp), set_shape(a, sp), set_shape(σ, sp2), set_shape(ε, sp2)
end