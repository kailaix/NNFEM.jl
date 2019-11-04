#!/bin/bash

# bgxupdate - update active processes in a group.
#   Works by transferring each process to new group
#   if it is still active.
# in:  bgxgrp - current group of processes.
# out: bgxgrp - new group of processes.
# out: bgxcount - number of processes in new group.

bgxupdate() {
    bgxoldgrp=${bgxgrp}
    bgxgrp=""
    ((bgxcount = 0))
    bgxjobs=" $(jobs -pr | tr '\n' ' ')"
    for bgxpid in ${bgxoldgrp} ; do
        echo "${bgxjobs}" | grep " ${bgxpid} " >/dev/null 2>&1
        if [[ $? -eq 0 ]] ; then
            bgxgrp="${bgxgrp} ${bgxpid}"
            ((bgxcount = bgxcount + 1))
        fi
    done
}

# bgxlimit - start a sub-process with a limit.

#   Loops, calling bgxupdate until there is a free
#   slot to run another sub-process. Then runs it
#   an updates the process group.
# in:  $1     - the limit on processes.
# in:  $2+    - the command to run for new process.
# in:  bgxgrp - the current group of processes.
# out: bgxgrp - new group of processes

bgxlimit() {
    bgxmax=$1 ; shift
    bgxupdate
    while [[ ${bgxcount} -ge ${bgxmax} ]] ; do
        sleep 1
        bgxupdate
    done
    if [[ "$1" != "-" ]] ; then
        $* &
        bgxgrp="${bgxgrp} $!"
    fi
}

group1=""
echo 0 $(date | awk '{print $4}') '[' ${group1} ']'
# echo
# for i in 1 2 3 4 5 6 ; do
#     bgxgrp=${group1} 
#     bgxlimit 4 sleep $i
#     group1=${bgxgrp}
#     echo ${i} $(date | awk '{print $4}') '[' ${group1} ']'
# done

echo
for force_scale in 5.0
do
for tid in 100 200 201 202 203 300
do
for porder in 2
do
for fiber_size in 5
do
bgxgrp=${group1} 
bgxlimit 6 julia Data_NNPlatePull.jl $tid $force_scale $fiber_size $porder > Debug/logs/${tid}_${force_scale}_${fiber_size}_$porder.txt 
group1=${bgxgrp}
echo "julia Data_NNPlatePull.jl $tid $force_scale $fiber_size $porder"
done
done
done
done
# 0.1 0.15 0.2

