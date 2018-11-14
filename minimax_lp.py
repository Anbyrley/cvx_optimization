import numpy
from cvxpy import *

'''
		minimize ||x||_inf
		subject to Ax = b

	Done by 1st order projected subgradient descent
'''

#===Set Seed===#
numpy.random.seed(42);

#===Make Constraint Matrix===#
M = 5; N = 3;
A = numpy.asmatrix(numpy.random.normal(0, 1.0, (M, N)));

#===Make b===#
x_k = numpy.asmatrix(numpy.random.uniform(0, 1.0, N)).T;
e = numpy.asmatrix(numpy.random.uniform(0, 0.1, M)).T;
b = A*x_k;

#===Make Cost===#
z = numpy.asmatrix(numpy.random.normal(0, 1, M)).T;
s = numpy.asmatrix(numpy.random.uniform(0,1,N)).T
c = A.T*z + s

#===Create Variable===#
x = Variable(N);

#===Create Linear Cost===#
cost = Parameter(N);
cost.value = c;

#===Create Constraint Parameters===#
B = Parameter(M, N);
B.value = A
d = Parameter(M);
d.value = b

#===Create Constraint System===#
constraints = [B*x == d];

#so generating it this way, x_k is already the optimal solution
#we then need to 'find it again' using our 1st order minimax algorithm
#To do so, we need to project onto Bx == d
#It can be done analytically with the projection problem



obj = Minimize(pnorm(x,p=numpy.inf));
prob = Problem(obj, constraints);
prob.solve();

print "f* =", obj.value
xopt = prob.variables()[0].value;
print "x* =",xopt
print "xk =",x_k
	


'''
#===Create And Solve Problem===#
#obj = Minimize(0.5*quad_form(x, H) + c.T*x);
#obj = Minimize(quad_form(x, D) + c.T*x);
obj = Minimize((pnorm(D_12*x,2.0)**2.0) + c.T*x);
prob = Problem(obj, constraints);
prob.solve();

print "f* =", obj.value
xopt = prob.variables()[0].value;
print "x* =",xopt
'''

