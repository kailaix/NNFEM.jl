# Creating Mesh for FEM

NNFEM has a built-in mesh generation module, and users can  generate unstructured quadrilateral meshes directly from NNFEM. The backend is [Gmsh](https://gmsh.info/). In the following, we will present several examples to illustrate how to generate mesh using the mesh module. 

The NNFEM mesh module provides a set of functions to create the geometry:

* Create 0D Entities: [`addPoint`](@ref)
* Create 1D Entities: [`addLine`](@ref), [`addCircleArc`](@ref)
* Create 1D Closed Curve: [`addCurveLoop`](@ref), [`addDisk`](@ref)
* Create 2D Entities: [`addPlaneSurface`](@ref)

A common workflow for using the mesh module is 

1. Initialize a session by calling [`init_gmsh`](@ref).
2. Build planar suface from lower dimensional entities using the functions shown above.
3. Generate mesh by running [`finalize_gmsh`](@ref). A `.msh` file is created and the file path is returned. This file can be read by [`meshread`](@ref)


Here we show some examples:

## Making a Rectangle Mesh
```julia
using NNFEM
init_gmsh()
p1 = addPoint(0, 0)
p2 = addPoint(2, 0)
p3 = addPoint(2, 1)
p4 = addPoint(0, 1)
l1 = addLine(p1, p2)
l2 = addLine(p2, p3)
l3 = addLine(p3, p4)
l4 = addLine(p4, p1)
cl = addCurveLoop([l1,l2,l3,l4])
s = addPlaneSurface([cl])
# finalize_gmsh takes 0 or 1 argument. If the argument is `true`, then the mesh is shown. The default is false.
finalize_gmsh(true)
```

```@raw html
<center><img src="[image.jpg](https://github.com/ADCMEMarket/ADCMEImages/blob/master/NNFEM/mesh1.png?raw=true)" style="width: 50%/>​</center>
```


## Making a Rectangle Mesh with a Hole

```julia
using NNFEM

init_gmsh()
pts = [
    addPoint(0,0),
    addPoint(2,0),
    addPoint(2,1),
    addPoint(0,1),
    addPoint(0.5,0.5),
    addPoint(1.5,0.5),
    addPoint(1.0, 0.5)
]
rectangle = addCurveLoop([
    addLine(pts[1], pts[2]),
    addLine(pts[2], pts[3]),
    addLine(pts[3], pts[4]),
    addLine(pts[4], pts[1])
])
hole = addCurveLoop(
    [
        addLine(pts[5], pts[6]),
        addCircleArc(pts[6], pts[7],pts[5])
    ]
)
s = addPlaneSurface([rectangle, hole])
finalize_gmsh(true)
```

```@raw html
<center><img src="[image.jpg](https://github.com/ADCMEMarket/ADCMEImages/blob/master/NNFEM/mesh2.png?raw=true)" style="width: 50%/>​</center>
```

## Embed an Entity in the Surface

NNFEM also allows you to embed a point in a line or a surface, or embed a line in a surface.

```julia
using NNFEM

init_gmsh()
pts = [
    addPoint(0,0),
    addPoint(2,0),
    addPoint(2,1),
    addPoint(0,1),
    addPoint(0.3,1.0),
    addPoint(0.8,0.5)
]
rectangle = addPlaneSurface(
    [addCurveLoop([
        addLine(1,2),
        addLine(2,3),
        addLine(3,4),
        addLine(4,1)
    ])]
)
line = addLine(5,6)
embedLine([line], rectangle)
finalize_gmsh(true)
```


```@raw html
<center><img src="[image.jpg](https://github.com/ADCMEMarket/ADCMEImages/blob/master/NNFEM/mesh4.png?raw=true)" style="width: 50%/>​</center>
```

## Control Mesh Size

Note the third argument of [`addPoint`](@ref) can be used to specify the mesh size at the specific point. Another method is to use [`meshsize`](@ref).

```julia
using NNFEM

init_gmsh()
pts = [
    addPoint(0,0),
    addPoint(2,0),
    addPoint(2,1),
    addPoint(0,1)
]
rectangle_loop = addCurveLoop([
        addLine(1,2),
        addLine(2,3),
        addLine(3,4),
        addLine(4,1)
    ])
disk_loop = addCircle(1.0,0.5,0.3)
rectangle_with_hole = addPlaneSurface([rectangle_loop, -disk_loop])
# meshsize takes a string as argument, which is a C++ syntax function
meshsize("0.1 *((x-1.0)*(x-1.0) + (y-0.5)*(y-0.5))")
finalize_gmsh(true)
```

```@raw html
<center><img src="[image.jpg](https://github.com/ADCMEMarket/ADCMEImages/blob/master/NNFEM/mesh3.png?raw=true)" style="width: 50%/>​</center>
```

## Add Physical Group Names

We can use [`addPhysicalGroup`](@ref) to add physical groups. 

```julia
using NNFEM

init_gmsh()
pts = [
    addPoint(0,0),
    addPoint(2,0),
    addPoint(2,1),
    addPoint(0,1),
    addPoint(0.3,1.0),
    addPoint(0.8,0.5)
]
rectangle = addPlaneSurface(
    [addCurveLoop([
        addLine(1,2),
        addLine(2,3),
        addLine(3,4),
        addLine(4,1)
    ])]
)
line = addLine(5,6)
addPhysicalGroup(1, line, "Neunmann")
addPhysicalGroup(0, [pts[end]], "Neunmann")
addPhysicalGroup(0, [6,pts[1]], "Dirichlet")
embedLine([line], rectangle)
p = finalize_gmsh(false)
```

Once we created the mesh file, we can read the physical group using 

```julia
psread(p)
```

Expected output:
```
Dict{String,Array{Float64,2}} with 2 entries:
  "Dirichlet" => [0.0 0.0]
  "Neunmann"  => [0.8 0.5; 0.3625 0.9375; … ; 0.55 0.75; 0.675 0.625]
```