# Dynamic Problems

NNFEM can be used to solve the folowing dynamical problem

$$\ddot u - {\text{div}}\sigma  = f, x\in \Omega \tag{1}$$

where $u$ is the 2D displacement vector, $\sigma$ is the stress, and $f$ is the body force. The dynamical equation is supplemented with two kinds of boundary conditions: Dirichlet boundary conditions and Neumann boundary conditions. For each type of conditions, we consider two types: time-dependent and time-independent. The following matrix shows all possible boundary conditions supported by NNFEM. 

|                  | Dirichlet                              | Neumann                                       |
| ---------------- | -------------------------------------- | --------------------------------------------- |
| Time-independent | $$u(x,t) = u_1(x), x\in \Gamma_D^1$$   | $$\sigma(x,t)n(x) = t_1(x), x\in \Gamma_N^1$$ |
| Time-dependent   | $$u(x,t) = u_2(x,t), x\in \Gamma_D^2$$ | $$\sigma(x,t)n(x) = t_2(x), x\in \Gamma_N^2$$ |

The weak form of Equation 1 can be written as 

$$\int_\Omega u \delta u dx  + \int_\Omega \sigma :\delta \epsilon dx = \int_\Omega f \delta u dx + \int_{\Gamma_N} t \delta u dx \tag{2}$$

Here 

$$\int_{\Gamma_N} t \delta u dx =\int_{\Gamma_N^1} t_1 \delta u dx + \int_{\Gamma_N^2} t_2 \delta u dx $$

In NNFEM, the boundary information are marked in `EBC` and `FBC` arrays in the geometry information `Domain`, respectively. These arrays have size $n_v\times 2$, corresponding to $n_v$ nodes and $x$-/$y$-directions. $-1$ represents time-independent boundaries and $-2$ represents time-dependent boundaries. Time indepdent boundary conditions `g` and `fext` are precomputed and fed to `Domain`, while time independent bounary conditions can be evaluated online with `EBC_func` and `FBC_func` in `GlobalData`. In the case the external load is provided as $t(x,t) = \sigma(x,t)n(x)$, we can use `Edge_func` and `Edge_Traction_Data` to provide the information instead of  `FBC_func`. 

If we express Equation 2 in terms of matrices, we have

$$M \ddot{\mathbf{u}} + K (\mathbf{u}) = \mathbf{f} + \mathbf{t}$$

Here $K(\mathbf{u})$ can be nonlinear. 

There are two solver implemented in NNFEM: the explicit solver and the generalized alpha solver. Both solvers support automatic differentiation for a linear $K$. The explicit solver also supports automatic differentiation for nonlinear $K$. 

To get started, you can study the following examples. Most likely you only need to modify the script to meet your own needs. 

