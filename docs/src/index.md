# Getting Started

## Motivation

Plasticity material has been described with different strain-stress relation models. However, these models are usually derived _ad hoc_ and may not reflect the plasticity properties of new materials. The models describe the so-called strain-stress relations; for plasticity strain-stress relations, the current stress depends on the historic strains and stresses. 

Data-driven approaches to discover the strain-stress relations are promising to produce more accurate phenomenonial plasticity models. However, one limitation to the data we can collect is that **stress are usually not directly observable**. Therefore, obvious machine learning techniques that use curve-fitting methods to reconstruct the strain-stress relations (i.e., strain as inputs, stress as outputs) are not applicable in practice. 

The mission of NNFEM is to solve this types of problem by combining finite element analysis and deep neural networks for data-driven modeling of plasticity.

## Problem Formulation

The observation data are time series of the displacements Gaussian points of each finite element in a 2D plasticity material
$$u_i^n\in \mathbb{R}, i=1,2,\ldots,n_{gp}$$
where $n$ denotes time step, $i$ denotes the index of $n_{gp}$ points.

Consequently the strain $\varepsilon_i^n$ at any time can deduced; however, the stress $\sigma_i^n$ is not observed. The task is to model the strain-stress relationship for the material
$$\sigma = f(\sigma_0, \varepsilon_0, \varepsilon, \Delta t)$$
where $\sigma_0$ and $\varepsilon_0$ are strain and stress at previous time step, $\varepsilon$ is the current strain, and $\Delta t$ is the time interval. $\sigma$ is the predicted strain. 

In addition, we assume we know the external force $f_{i,ext}^n$, which is usually designed by experimentalist. 

Our goal is to contruct a deep neural network model $f$ that can reproduces the displacements data $u_i^n$. Particularly, the following are a few important validations for our approach

- Loading and unloading should be predicted accurately by our model.
- Displacements with unseen external loads should be predicted by our model (generalization).
- Stress estimation should be not far from ground truth (accuracy).

## Methodology

Our methodology is to formulate the internal force with finite element methods
$$F_{j, int}^n=G(\{u_i^k\}_{k\leq n, i\leq n_{gp}}, w)$$
where $w$ is the weights of the neural network $f$. 

The contribution of the external force  to each element is also computed $F_{j, ext}^n$.

We solve the following minimization problem 
$$\min_{w} \sum_{n=1,2,\ldot, i=1,2,\ldots, n_{gp}}\left( F_{j, int}^n - F_{j, ext}^n \right)$$

