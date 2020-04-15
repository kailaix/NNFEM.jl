# Representations of Constitutive Relation



## Constitutive Theory

To characterize material mechanics, we usually need to provide relations between kinematic and dynamic quantities. For example, the balance of linear momentum provides a constraint on the material mechanics. Another important relation is the **constitutive relation**, which describes the material's response to deformation. To describe the constitutive relations, the strain and stress tensors are useful. The strain tensor is automatic symmetric. The stress tensor can be shown to be symmetric based on the balance of angular momentum. 

Mathematically, let $\mathcal{C}$ denote the constitutive quantity, then the most general constitutive relation for $\mathcal{C}$ is given by a functional of the form 

$$\mathcal{C}(X, t) = \mathcal{F}_{Y\in \mathcal{B}, -\infty < s\leq  t} (\rho(Y, s), \chi(Y, s), \theta(Y, s), Y, t)$$

Here $\mathcal{F}$ is the constitutive function, $\mathcal{B}$ is the region of influence for $X$. $\rho$, $\chi$, $\theta$ are te density, the motion, and the temperature. For pure mechanical processes, the constitutive quantity $\mathcal{C}$ can be the stress tensor, the heat flux or the internal energy. 

Not all relations are valid constitutive relations. In the following, we consider some common assumptions on the constitutive relation. 

* **Principle of Determinism**. This means the current state of a material is determined tthrough the current motion and the entire motion-history of all other material points of the continuum body. 

* **Principle of Material Objectivity**. This is also known as observer or frame indifference. A material equation must not depend on the choice of the reference frame or observer. An observer deduces a physical law in his or her coordinate $\mathbf{x}$, and another observer, who is in another coordinate system $\mathbf{x}^*$, deduces another physical law for the same physical quantities. 

$$\mathbf{x}^* = Q(t) \mathbf{x} + \mathbf{x}(t), \qquad Q(t)Q(t)^T = Q(t)^TQ(t)=\mathbf{I}, \mathrm{det}(Q(t)) = \pm 1$$

Theses two laws should be consistent with each other. For example, the law might be the Newton's second law, which states that the acceleration is proportional to the external force. 

* **Non-local Materials**. The physical action at the point $X$ is determined by the action at all other points $Y$ of the body, and the region of influence is only restricted to the neighborhood of $X$. This assumption allows us to describe the physical law using the Taylor series of the state variables. For example, for the motion $\chi(Y, t)$, we have 

$$\chi(Y, t) = \chi(X, t) + \frac{1}{1!}\frac{\partial \chi(X, t)}{\partial \chi}\Delta X +\frac{1}{2!}(\Delta X)^T\frac{\partial^2 \chi(X, t)}{\partial \chi^2}\Delta X +\ldots$$

!!! note
    We can classify the material by the number of terms in the Taylor expansion above. A material is said to have **grade $N$** if its constitutive relation is described $N+1$ terms in the Taylor expansion. 

!!! note 
    Nonlocal constitutive relations, such as peridynamics, are also considered in literature. 

  
* **Material Symmetry**. In the case of material symmetries, we can further reduce the general constitutive relation. For example, for hyperelasticity materials, the material symmetry reduces the constitutive relation between strain and stress to isotropic functions. 


## Enforcing Physical Constraints in Constitutive Relations

In NNFEM, we provide a set of tools to enforce physical constraints. 

* Isotropic functions for on tensor

```@docs
isotropic_function
```

!!! info
    To construct a constitutive relation that has the form 

    ```math
    T = s_0(\sigma_1, \sigma_2)  + s_1(\sigma_1, \sigma_2) S  + s_2(\sigma_1, \sigma_2) S^2
    ```

    Here $\sigma_1$ and $\sigma_2$ are the two eigenvalues of $S$, we can use [`strain_voigt_to_tensor`](@ref) to extract eigenvalues.

    ```julia
    using NNFEM, ADCME
    strain = rand(100,3)
    strain_tensor = strain_voigt_to_tensor(strain)
    e, v = tf.linalg.eigh(strain_tensor)
    coef = ae(e, [20,20,20,3])
    stress = isotropic_function(coef, strain)
    ```


* Isotropic functions for two tensors
```@docs
bi_isotropic_function
```

Similar to the isotropic function for one tensor, we can construct a constitutive relation that is an isotropic function of two tensors. 
```julia
using NNFEM, ADCME
strain = rand(100,3)
strain_rate = rand(100,3)
strain_tensor = strain_voigt_to_tensor(strain)
strain_rate_tensor = strain_voigt_to_tensor(strain_rate)
e1, v = tf.linalg.eigh(strain_tensor)
e2, v = tf.linalg.eigh(strain_rate_tensor)
coef = ae([e1 e2], [20,20,20,9])
stress = isotropic_function(coef, strain)
```

* Plasticity 
```@docs
consistent_tangent_matrix
```

