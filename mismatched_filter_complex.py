import numpy
import matplotlib.pyplot as plt
from cvxpy import *

'''
		minimize [0, 1]^T [q, t]
		subject to s^H q = s^H s
					q^H l_i^H l_i q <= t, (i != N)
'''

#===Make Signal===#
N = 13;
s_original = numpy.random.uniform(-1.0, 1.0, N) + 1j * numpy.random.uniform(-1.0, 1.0, N);
s_original = numpy.array([1,1,1,1,1,-1,-1,1,1,-1,1,-1,1], dtype=complex);
s_vec = numpy.asmatrix(s_original).T;

#===Make Convolution Matrix===#
N = len(s_original);
L = numpy.zeros((2*N-1, N), dtype = complex);
r = 0;
for c in range(N):
	L[r:r+N, c] = s_original[::-1];
	r += 1;
L = numpy.asmatrix(L);

#===Make F Matrix===#
ones = numpy.r_[numpy.ones(N-1, dtype=complex), 0, numpy.ones(N-1, dtype=complex)];
F = numpy.asmatrix(numpy.diag(ones));

#===Make Middle Matrix===#
M_val = L.H * F * L

#===Create Variable===#
t_var = Variable(1);
q_var = Variable(N, complex=True);

#===Create Loss of Processing Gain Constraint===#
LPG = -10.0;
alpha = 10.0**(LPG/-10.0); 
I = numpy.identity(N, dtype=complex);

#===Create Constraint System===#
constraints = [real(quad_form(q_var,M_val)) <= t_var, s_vec.H*q_var == s_vec.H*s_vec, real(quad_form(q_var, I)) <= real(alpha * s_vec.H*s_vec)];

#===Solve System===#
obj = Minimize(t_var);
prob = Problem(obj, constraints);
prob.solve();

#===Extract===#
print "f0* =", obj.value
xopt = prob.variables()[1].value;
print "q* =",xopt
q = numpy.ravel(xopt);


#===Convolve===#
rss = numpy.convolve(s_original, numpy.conjugate(s_original)[::-1]);
#rss = numpy.abs(rss);
rss = rss.real;
csq = numpy.convolve(s_original, numpy.conjugate(q)[::-1]);
#csq = numpy.abs(csq);
csq = csq.real;
print "s*s = ", rss
print "s*q = ", csq

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
xopt = numpy.asmatrix(xopt).T;
print "Ending LPG: ", 10.0*numpy.log10((s_vec.conj().T*s_vec)/(xopt.conj().T*xopt));

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
