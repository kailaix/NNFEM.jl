for mode in linear consistent_tangent nn free_energy
do 
rm -rf $mode 
srun julia hyperelasticity_nn.jl $mode  & 
done
wait 