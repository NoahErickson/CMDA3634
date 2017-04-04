#! /bin/bash
#
#PBS -l walltime=00:05:00
#PBS -l nodes=8;ppn=1
#PBS -W group_list=hokiespeed
#PBS -q normal_q
#PBS -j oe

cd $PBS_O_WORKDIR

mpirun -n 1 ./conway test.txt
./conway
