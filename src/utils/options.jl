export Options
"""
    Options

Various options for NNFEM simulation. 

- `save_history::Int64`

  - 0: no histroy is saved 
  - 1: save stress, displacement, acceleration history 
  - 2: in addition to 1, save internal and external force 
"""
mutable struct Options
    save_history::Int64
end


function Options()
    Options(2)
end

options = Options()