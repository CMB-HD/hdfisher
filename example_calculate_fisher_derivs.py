"""Example for calculating the numerical derivatives of the CMB and BAO
signal with respect to the cosmological parameters, using the `hdfisher`
defaults.

"""

import os
import argparse
import warnings
from hdfisher import fisher, mpi

# command-line argument for using CLASS instead of CAMB:
description = ("Calculate the numerical derivatives of the CMB and BAO "
               "signal with respect to the cosmological parameters, "
               "using the `hdfisher` defaults.")
formatter_class = argparse.ArgumentDefaultsHelpFormatter
parser = argparse.ArgumentParser(formatter_class=formatter_class, description=description)
parser.add_argument('--use_class', action='store_true',
                    help="Pass `--use_class` to use CLASS instead of CAMB.")
args = parser.parse_args()

if args.use_class:
    msg = ("NOTE that CLASS must be modified to use the default set of CLASS "
           "parameters. If you have already done so, please ignore this message. "
           "Otherwise, follow the instructions given in the `hdfisher` README file.")
    warnings.warn(msg)

# set the `fisher_dir` that holds the output:
fisher_root = 'hd_example_CLASS' if args.use_class else 'hd_example'
output_dir = os.path.join(os.getcwd(), 'example_output')
fisher_dir = os.path.join(output_dir, fisher_root)

# initialize the `Fisher` class:
fisherlib = fisher.Fisher(fisher_dir, use_class=args.use_class)
# calculate two sets of derivatives:
for use_H0 in [False, True]:
    fisherlib.calculate_fisher_derivs(use_H0=use_H0)
    mpi.comm.barrier()

