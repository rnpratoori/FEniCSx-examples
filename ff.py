import sys
sys.path.append('/home/rnp/MOOSE/projects/moose/python')

import mms
fs,ss = mms.evaluate('-div(grad(u)) - lambda_ * exp(u)', 'sin(pi*x)*sin(pi*y)', variable='u', scalars=['lambda_'])
mms.print_fparser(fs)

mms.print_hit(ss, 'exact')
mms.print_hit(fs, 'force')