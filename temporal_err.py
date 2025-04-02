import sys
sys.path.append('/home/rnp/MOOSE/projects/moose/python')

import mms
# df1 = mms.run_temporal('tdht.i', 8, console=False, executable='/home/rnp/MOOSE/projects/moose/test/', csv='results/tdht_err.csv')
df2 = mms.run_temporal('tdht.i', 8, 'Executioner/scheme=CrankNicolson',
                      console=False, executable='/home/rnp/MOOSE/projects/moose/test/', csv='results/tdht_err_t.csv')

fig = mms.ConvergencePlot(xlabel=r'$\Delta$t', ylabel='$L_2$ Error')
# fig.plot(df1, label=['1st Order (Implicit Euler)'], marker='o', markersize=8)
fig.plot(df2, label=['2nd Order'], marker='o', markersize=8)  # List of labels
fig.save('tdht_err_t.png')