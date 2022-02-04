# Q&A

* Q: How can I get the number of Dirichlet boundary nodes (including directions)?

```julia
sum(domain.EBC[:].!=0)
```