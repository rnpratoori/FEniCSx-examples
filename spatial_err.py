import sys
from mpi4py import MPI
sys.path.append('/home/rnp/MOOSE/projects/moose/python')

import mms
df1 = mms.run_spatial('tdht.i', 8, console=False, executable='/home/rnp/MOOSE/projects/moose/test/', file_base='results/tdht_err')
# df2 = mms.run_spatial('tdht.i', 8, 'Mesh/second_order=true', 'Variables/u/order=SECOND',
#                       console=False, executable='/home/rnp/MOOSE/projects/moose/test/')

if MPI.COMM_WORLD.Get_rank() == 0:
    fig = mms.ConvergencePlot(xlabel='Element Size ($h$)', ylabel='$L_2$ Error')
    fig.plot(df1, label=['1st Order'], marker='o', markersize=8)  # List of labels
    # fig.plot(df2, label=['2nd Order'], marker='o', markersize=8)  # List of labels
    fig.save('tdht_err.png')