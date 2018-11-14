from cvxopt import matrix, solvers

'''
	minimize 2x1 + x2
	subject to -x1 + x2 <= 1
				x1 + x2 >= 2
				x2 >= 0
				x1 - 2x2 <= 4
'''

A = matrix([ [-1.0, -1.0, 0.0, 1.0], [1.0, -1.0, -1.0, -2.0] ])
b = matrix([ 1.0, -2.0, 0.0, 4.0 ])
c = matrix([ 2.0, 1.0 ])

print "Hello!"

print A
print b
print c

sol=solvers.lp(c,A,b)

print sol['x'];
