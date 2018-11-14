import numpy
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers

'''
	Saramaki says that we only need to constrain the stopband
		-- but this gives me a bad filter
	Also can reduce size of KKT matrix via Saramaki's 'Subtraction' technique?
		-- Might wanna try this.
'''

#===Set Up Parameters===#
L = 2.0;
filter_length = 115; #55 works for 0.4
M = (filter_length-1)/2;
num_nodes = int(15*filter_length);
filter_order = filter_length-1;

#===Create Frequency Nodes===#
rolloff = 0.15;
omega = numpy.pi/float(num_nodes) * numpy.linspace(0, num_nodes, num_nodes); 
omega_pass = (1.0-rolloff) * (1.0/L) * numpy.pi; 
omega_stop = (1.0+rolloff) * (1.0/L) * numpy.pi; 

#===Form Bands===#
if (1):

	#===Form Passband===#
	passband = [];
	if (1):
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
pass_weight = 1.0;
stop_weight = 500.0;
Weights = numpy.zeros(num_nodes);
for w in range(num_nodes):
	if (omega[w] <= omega_pass):
		Weights[w] = pass_weight;
	elif (omega[w] >= omega_stop):
		Weights[w] = stop_weight;
	else:
		Weights[w] = 10e-16;

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
Block = (numpy.r_[C, mC]);

#===Form Identity Matrix===#
Identity = numpy.identity(M+1);
Zeros = numpy.matrix(numpy.zeros(M+1)).T;
I_tilde = numpy.c_[Identity, Zeros];
mI_tilde = -1.0*I_tilde;

#===Form U Vector===#
constant = 1.0;
u = numpy.matrix(numpy.zeros(M+1)).T;
u[0] = 1.0/L;
for r in range(1,u.shape[0]):
	if (r%int(L)==0):
		u[r] = 0.0;
	else:
		u[r] = constant;

#===Form l Vector===#
l = numpy.matrix(numpy.zeros(M+1)).T;
l[0] = -1.0/L;
for r in range(1,l.shape[0]):
	if (r%L==0):
		l[r] = -0.0;
	else:
		l[r] = -1.0*constant;

#===Append Box Constraints To Block Matrix===#
New_Block = numpy.r_[I_tilde, mI_tilde];
Block = numpy.r_[Block, New_Block];
Block = matrix(Block);

#===Form RHS Vector===#
d = numpy.asmatrix(numpy.r_[Desired, -Desired]).T;
d = numpy.r_[d, u, -l];
d = matrix(d);

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
plt.ylim([1.1*numpy.amin(logmag), 2.0]);

indices = numpy.arange(0, len(design));
plt.figure(1);
markerline, stemlines, baseline = plt.stem(indices, design, '-.')
plt.setp(baseline, 'color', 'r', 'linewidth', 2)
#plt.plot(design);
plt.title("LP Time Domain");

plt.show();
