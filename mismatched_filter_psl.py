import numpy
import matplotlib.pyplot as plt
from cvxpy import *

'''
		minimize [0, 1]^T [q, t]
		subject to s^H q = s^H s
					q^H l_i^H l_i q <= t, (i != N)
'''

#===Make Signal===#
s = [1.0, 0.75, -1.0, 0.5];
s_vec = numpy.asmatrix(s).T;

#N = 32;
#s = numpy.random.uniform(-1.0, 1.0, N);
s = [1,1,1,1,1,-1,-1,1,1,-1,1,-1,1];
N = 13;
s_vec = numpy.asmatrix(s).T;


#===Make Convolution Matrix===#
N = len(s);
L = numpy.zeros((2*N-1, N), dtype = float);
r = 0;
for c in range(N):
	L[r:r+N, c] = s[::-1];
	r += 1;
L = numpy.asmatrix(L);

#===Create Variable===#
t_var = Variable(1);
q_var = Variable(N);

#===Create Loss of Processing Gain Constraint===#
LPG = -3.0;
alpha = 10.0**(LPG/-10.0); 
I = numpy.asmatrix(numpy.identity(N));

#===Make Constraint System===#
matrices = [];
for i in range(L.shape[0]):
	if (i != 3):
		row = L[i, :];
		matrix = row.T*row;
		matrix += 10e-16 * numpy.identity(matrix.shape[0]); #ensure psd
		matrices.append(matrix);
constraints = [];
for i in range(len(matrices)):
	constraints.append( quad_form(q_var, matrices[i]) <= t_var );
constraints.append(s_vec.T*q_var == s_vec.T*s_vec);
constraints.append(quad_form(q_var, I) <= alpha * s_vec.T*s_vec);

#===Solve System===#
obj = Minimize(t_var);
prob = Problem(obj, constraints);
prob.solve();

#===Extract===#
print "f0* =", obj.value
xopt = prob.variables()[1].value;
print "q* =",xopt
q = numpy.ravel(xopt);
q_vec = numpy.asmatrix(q).T;

#===Convolve===#
rss = numpy.convolve(s, s[::-1]);
csq = numpy.convolve(s, q[::-1]);
print "s*s = ", rss
print "s*q = ", csq

#===Make F Matrix===#
ones = numpy.r_[numpy.ones(N-1), 0, numpy.ones(N-1)];
F = numpy.asmatrix(numpy.diag(ones));

#===Print M/S Ratio Improvement===#
sidelobes = F*numpy.asmatrix(rss).T; mainlobe = numpy.asmatrix(rss).T - sidelobes;
print "Matched Mainlobe to Sidelobe Ratio (dB): ", 10.0*numpy.log10(numpy.amax(mainlobe)/numpy.amax(sidelobes));

sidelobes = F*numpy.asmatrix(csq).T; mainlobe = numpy.asmatrix(csq).T - sidelobes;
print "Mismatched Mainlobe to Sidelobe Ratio (dB): ", 10.0*numpy.log10(numpy.amax(mainlobe)/numpy.amax(sidelobes));

#===Setup Plot===#
samples = numpy.linspace(-(len(rss)-1)/2, (len(rss)-1)/2, len(rss));
minn = numpy.amin([numpy.amin(rss), numpy.amin(csq)]);
maxx = numpy.amax([numpy.amax(rss), numpy.max(csq)]);
print "Max Difference (%): ", 100.0* numpy.amax(numpy.abs(rss - csq))/maxx;
print "Designed LPG: ", -10.0*numpy.log10(alpha);
print "Ending LPG: ", 10.0*numpy.log10(numpy.ravel(s_vec.T*s_vec)[0]/numpy.ravel(q_vec.T*q_vec)[0]);

#===Plot===#
font_size = 16;
fig0 = plt.figure(0);
ax0 = plt.subplot(111);
markerline, stemlines, baseline = ax0.stem(samples, rss, '--')
ax0.set_title("Rss", fontsize=font_size);
ax0.set_ylabel("Correlation Value");
ax0.set_xlabel("Sample");
ax0.yaxis.label.set_size(font_size)
ax0.xaxis.label.set_size(font_size)
ax0.tick_params(axis='x', which='major', labelsize=15)
ax0.tick_params(axis='y', which='major', labelsize=15)
ax0.set_xlim([samples[0]-1, samples[-1]+1]);
ax0.set_ylim([1.1*minn, 1.1*maxx]);
plt.setp(markerline, 'markerfacecolor', 'b');
plt.setp(baseline, 'color','b', 'linewidth', 2);

#fig1 = plt.figure(1);
ax1 = plt.subplot(111);
markerline, stemlines, baseline = ax1.stem(samples, csq, '--')
ax1.set_title("Csq", fontsize=font_size);
ax1.set_ylabel("Correlation Value");
ax1.set_xlabel("Sample");
ax1.yaxis.label.set_size(font_size)
ax1.xaxis.label.set_size(font_size)
ax1.tick_params(axis='x', which='major', labelsize=15)
ax1.tick_params(axis='y', which='major', labelsize=15)
ax1.set_xlim([samples[0]-1, samples[-1]+1]);
ax1.set_ylim([1.1*minn, 1.1*maxx]);
plt.setp(markerline, 'markerfacecolor', 'g');
plt.setp(baseline, 'color','g', 'linewidth', 2);

plt.show();

