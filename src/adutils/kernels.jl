
export s_eval_strain_on_gauss_points, s_compute_stiffness_matrix,
s_compute_internal_force_term,
f_eval_strain_on_gauss_points, f_compute_internal_force_term,
s_compute_stiffness_matrix1




@doc raw"""
    s_compute_stiffness_matrix(k::Union{Array{Float64,3}, PyObject})

Computes the small strain stiffness matrix. $k$ is a $n\times 3\times 3$ matrix, where $n$ is the total number of Gauss points.
Returns a SparseTensor. 
"""
function s_compute_stiffness_matrix(k::Union{Array{Float64,3}, PyObject}, domain::Domain)
    small_continuum_stiffness_ = load_op_and_grad("$(@__DIR__)/../../deps/CustomOp/build/libDataLib","small_continuum_stiffness", multiple=true)
    k = convert_to_tensor([k], [Float64]); k = k[1]
    ii, jj, vv = small_continuum_stiffness_(k)
    SparseTensor(ii+1, jj+1, vv, domain.neqs, domain.neqs)
end


@doc raw"""
    s_eval_strain_on_gauss_points(state::Union{Array{Float64,1}, PyObject})

Computes the strain on Gauss points in the small strain case. `state` is the full displacement vector. 
"""
function s_eval_strain_on_gauss_points(state::Union{Array{Float64,1}, PyObject}, domain::Domain)
    small_continuum_strain_ = load_op_and_grad("$(@__DIR__)/../../deps/CustomOp/build/libDataLib","small_continuum_strain")
    state = convert_to_tensor([state], [Float64]); state = state[1]
    ep = small_continuum_strain_(state)
    set_shape(ep, (getNGauss(domain), 3))
end

@doc raw"""
    f_eval_strain_on_gauss_points(state::Union{Array{Float64,1}, PyObject})

Computes the strain on Gauss points in the finite strain case. `state` is the full displacement vector. 
"""
function f_eval_strain_on_gauss_points(state::Union{Array{Float64,1}, PyObject}, domain::Domain)
    finit_continuum_strain_ = load_op_and_grad("$(@__DIR__)/../../deps/CustomOp/build/libDataLib","finit_continuum_strain")
    state = convert_to_tensor([state], [Float64]); state = state[1]
    ep = finit_continuum_strain_(state)
    set_shape(ep, (getNGauss(domain), 3))
end



@doc raw"""
    s_compute_internal_force_term(stress::Union{Array{Float64,2}, PyObject})

Computes the internal force
$$\int_\Omega \sigma : \delta \epsilon dx$$
Only active DOFs are considered. 
"""
function s_compute_internal_force_term(stress::Union{Array{Float64,2}, PyObject}, domain::Domain)
    small_continuum_fint_ = load_op_and_grad("$(@__DIR__)/../../deps/CustomOp/build/libDataLib","small_continuum_fint")
    stress = convert_to_tensor([stress], [Float64]); stress = stress[1]
    out = small_continuum_fint_(stress)
    set_shape(out, (domain.neqs,))
end


@doc raw"""
    f_compute_internal_force_term(stress::Union{Array{Float64,2}, PyObject}, 
        state::Union{Array{Float64,1}, PyObject},
        domain::Domain)

Computes the internal force for finite strain continuum

$$\int_\Omega \sigma : \delta \epsilon dx$$

Only active DOFs are considered. 
"""
function f_compute_internal_force_term(stress::Union{Array{Float64,2}, PyObject}, 
    state::Union{Array{Float64,1}, PyObject},
    domain::Domain)
    finite_continuum_fint_ = load_op_and_grad("$(@__DIR__)/../../deps/CustomOp/build/libDataLib","finite_continuum_fint")
    stress,state = convert_to_tensor([stress,state], [Float64,Float64])
    out = finite_continuum_fint_(stress,state)
    set_shape(out, (domain.neqs,))
end



@doc raw"""
    s_compute_stiffness_matrix1(k::Union{PyObject, Array{Float64,3}}, domain::Domain)

Computes the stiffness matrix 
```
\int_\Omega (K\nabla u) \cdot \nabla \delta u dx 
```
"""
function s_compute_stiffness_matrix1(k::Union{PyObject, Array{Float64,3}}, domain::Domain)
    small_continuum_stiffness1_ = load_op_and_grad("$(@__DIR__)/../../deps/CustomOp/build/libDataLib","small_continuum_stiffness1", multiple=true)
    k = convert_to_tensor(Any[k], [Float64]); k = k[1]
    ii, jj, vv = small_continuum_stiffness1_(k)
    A = SparseTensor(ii+1, jj+1, vv, domain.neqs, domain.neqs)
end
