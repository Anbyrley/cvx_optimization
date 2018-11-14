import numpy
import matplotlib.pyplot as plt
from cvxpy import *

'''
		minimize [0, 1]^T [q, t]
		subject to s^H q = s^H s
					q^H l_i^H l_i q <= t, (i != N)
'''


#===Make Signal===#
p = 0;
N = 32; K = N+2*p;
if (K % 2 == 0): K += 1;
s = [1,1,1,1,1,-1,-1,1,1,-1,1,-1,1];
s = numpy.random.uniform(-1.0, 1.0, N);
#s = [1,1,-1,1,-1];
s_original = numpy.copy(s);
s = numpy.r_[s, numpy.zeros(K-N)];
s_vec = numpy.asmatrix(s).T;
	

#===Make Convolution Matrix===#
L = numpy.zeros((K+N-1, K), dtype = float);
r = 0;
for c in range(K):
	L[r:r+N, c] = s_original[::-1];
	r += 1;
L = numpy.asmatrix(L);

#===Make F Matrix===#
ones = numpy.ones(K+N-1);
ones[N+p-1] = 0.0;
F = numpy.asmatrix(numpy.diag(ones));

#===Make Other F===#
ones = numpy.r_[numpy.ones(N-1), 0, numpy.ones(N-1)];
F_rss = numpy.asmatrix(numpy.diag(ones));

#===Make Middle Matrix===#
M_val = L.T * F * L

#===Create Variable===#
t_var = Variable(1);
q_var = Variable(K);

#===Create Loss of Processing Gain Constraint===#
LPG = -3.5;
alpha = 10.0**(LPG/-10.0); 
I = numpy.asmatrix(numpy.identity(K));

#===Create Constraint System===#
constraints = [quad_form(q_var,M_val) <= t_var, s_vec.T*q_var == s_vec.T*s_vec, quad_form(q_var, I) <= alpha * s_vec.T*s_vec];

#===Solve System===#
obj = Minimize(t_var);
prob = Problem(obj, constraints);
prob.solve();

#===Extract===#
print "f0* =", obj.value
xopt = prob.variables()[1].value;
print "q* =",xopt
q = numpy.ravel(xopt);
q = q[0:N];

#===Convolve===#
rss = numpy.convolve(s_original, s_original[::-1]);
csq = numpy.convolve(s_original, q[::-1]);
print "s*s = ", rss
print "s*q = ", csq

#===Print M/S Ratio Improvement===#
sidelobes = F_rss*numpy.asmatrix(rss).T; mainlobe = numpy.asmatrix(rss).T - sidelobes;
print "Matched Mainlobe to Sidelobe Ratio (dB): ", 10.0*numpy.log10(numpy.amax(mainlobe)/numpy.amax(sidelobes));

#sidelobes = F*numpy.asmatrix(csq).T; mainlobe = numpy.asmatrix(csq).T - sidelobes;
sidelobes = F_rss*numpy.asmatrix(csq).T; mainlobe = numpy.asmatrix(csq).T - sidelobes;
print "Mismatched Mainlobe to Sidelobe Ratio (dB): ", 10.0*numpy.log10(numpy.amax(mainlobe)/numpy.amax(sidelobes));

#===Setup Plot===#
rss_samples = numpy.linspace(-(len(rss)-1)/2, (len(rss)-1)/2, len(rss));
csq_samples = numpy.linspace(-(len(csq)-1)/2, (len(csq)-1)/2, len(csq));
minn = numpy.amin([numpy.amin(rss), numpy.amin(csq)]);
maxx = numpy.amax([numpy.amax(rss), numpy.max(csq)]);
print "Designed LPG: ", -10.0*numpy.log10(alpha);
print "Ending LPG: ", 10.0*numpy.log10((s_vec.T*s_vec)/(xopt.T*xopt));

#===Plot===#
font_size = 16;
fig0 = plt.figure(0);
ax0 = plt.subplot(111);
markerline, stemlines, baseline = ax0.stem(rss_samples, rss, '--')
ax0.set_title("Rss", fontsize=font_size);
ax0.set_ylabel("Correlation Value");
ax0.set_xlabel("Sample");
ax0.yaxis.label.set_size(font_size)
ax0.xaxis.label.set_size(font_size)
ax0.tick_params(axis='x', which='major', labelsize=15)
ax0.tick_params(axis='y', which='major', labelsize=15)
ax0.set_xlim([rss_samples[0]-1, rss_samples[-1]+1]);
ax0.set_ylim([1.1*minn, 1.1*maxx]);
plt.setp(markerline, 'markerfacecolor', 'b');
plt.setp(baseline, 'color','b', 'linewidth', 2);

fig1 = plt.figure(1);
ax1 = plt.subplot(111);
markerline, stemlines, baseline = ax1.stem(csq_samples, csq, '--')
ax1.set_title("Csq", fontsize=font_size);
ax1.set_ylabel("Correlation Value");
ax1.set_xlabel("Sample");
ax1.yaxis.label.set_size(font_size)
ax1.xaxis.label.set_size(font_size)
ax1.tick_params(axis='x', which='major', labelsize=15)
ax1.tick_params(axis='y', which='major', labelsize=15)
ax1.set_xlim([csq_samples[0]-1, csq_samples[-1]+1]);
ax1.set_ylim([1.1*minn, 1.1*maxx]);
plt.setp(markerline, 'markerfacecolor', 'g');
plt.setp(baseline, 'color','g', 'linewidth', 2);

plt.show();
