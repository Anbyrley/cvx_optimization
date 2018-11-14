import numpy
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers

'''
	So it appears that it is not very good in the passband
	But pretty good in the stopband
	So that is how it should be designed
		-- need a large filter order, but this sometimes makes it singular
	So the omega_pass needs to be lessened and adjusted to fit targets
	num_nodes about 15*filter_length
'''

#===Set Up Parameters===#
filter_length = 55; #55 works for 0.4
M = (filter_length-1)/2;
num_nodes = int(15*filter_length);
filter_order = filter_length-1;

#===Create Frequency Nodes===#
omega = numpy.pi/float(num_nodes) * numpy.linspace(0, num_nodes, num_nodes); 
omega_pass = 0.35 * numpy.pi; #this works for 0.4 passband
omega_stop = 0.5 * numpy.pi; #this works for 0.5 passband

#===Form Bands===#
if (1):
	#===Form Passband===#
	passband = [];
	w = 0;
	while(omega[w] <= omega_pass):
		passband.append(omega[w]);
		w = w+1;

	#===Form Stopband===#
	stopband = [];
	w = 0;
	for w in range(num_nodes):
		if (omega[w] >= omega_stop):
			stopband.append(omega[w]);

	#===Concatenate===#
	omega = numpy.r_[passband, stopband];
	num_nodes = len(omega);
print "Num Nodes", num_nodes;

#===Form Desired Response===#
w = 0; 
Desired = 10e-8*numpy.ones(num_nodes);
while(omega[w] <= omega_pass):
	Desired[w] = 1.0;
	w = w + 1;

#===Form Weight Vector===#
pass_weight = 500.0;
stop_weight = 50.0;
transition_weight = 10.0;
Weights = numpy.zeros(num_nodes);
for w in range(num_nodes):
	if (omega[w] <= omega_pass):
		Weights[w] = pass_weight;
	elif (omega[w] >= omega_stop):
		Weights[w] = stop_weight;
	else:
		Weights[w] = transition_weight;

#===Form Type 1 Linear Phase Matrix Column Wise===#
C = numpy.asmatrix(numpy.ones(num_nodes)).T;
for c in range(1,M+1):
	column = [numpy.cos(c * omega[l]) for l in range(num_nodes)];
	C = numpy.c_[C, column];
mC = -1.0*C;

#===Append Inverse Weight Vector===#
C = numpy.c_[C,-1.0/Weights];
mC = numpy.c_[mC, -1.0/Weights];

#===Form Block Matrix===#
Block = matrix(numpy.r_[C, mC]);

#===Form RHS Vector===#
d = matrix(numpy.asmatrix(numpy.r_[Desired, -Desired]).T);

#===Form Cost Vector===#
cost = matrix(numpy.r_[numpy.zeros(M+1), 1]);

#===Solve LP===#
sol=solvers.lp(cost,Block,d)

#===Parse Solution===#
a = sol['x'][0:-1];
delta = sol['x'][-1];
print "Delta: ", delta

#===Make Filter===#
design = list(a[-1:0:-1]);
design.append(2.0*a[0]);
temp = list(a[1::]);
for t in temp:
	design.append(t);
print "Filter Length: ", len(design), filter_length;

#===Ensure Unity Gain===#
design = numpy.array(design);
design /= sum(design);

#===Compute Frequency Response===#
fft = numpy.fft.fftshift(numpy.fft.fft(design));
freqs = numpy.fft.fftshift(numpy.fft.fftfreq(len(fft), d=0.5));
mag = abs(fft)**2.0;
logmag = 20.0*numpy.log10(mag);

#===Plot===#
plt.figure(0);
plt.plot(freqs, logmag);
plt.title("LP Frequency Response");
plt.ylim([1.1*numpy.amin(logmag), 1.0]);
plt.show();
