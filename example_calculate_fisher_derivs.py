import os
from hdfisher import fisher, mpi

# set the `fisher_dir` that holds the output:
fisher_root = 'hd_example'
output_dir = os.path.join(os.getcwd(), 'example_output')
fisher_dir = os.path.join(output_dir, fisher_root)


for use_H0 in [False, True]:
    fisherlib = fisher.Fisher(fisher_dir, use_H0=use_H0)
    fisherlib.calculate_fisher_derivs()
    mpi.comm.barrier()
