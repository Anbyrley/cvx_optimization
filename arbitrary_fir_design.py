import numpy
import scipy.signal as signal
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers

#===Set Up Parameters===#
filter_length = 37;
M = (filter_length-1)/2;
num_nodes = int(15*filter_length);
filter_order = filter_length-1;

#===Make Desired Signal===#
use_scipy = 1;
if (use_scipy):
	passband = 0.25;
	stopband = 0.35;
	h_desired = signal.remez(filter_length, [0.0, passband, stopband, 0.5], [0.0, 1.0], Hz=1.0);
	#h_desired = signal.remez(filter_length, [0.0, 0.25, 0.35, 0.5], [1.0, 1.0], Hz=1.0);
	desired = numpy.fft.fftshift(numpy.fft.fft(h_desired, n=num_nodes));
	omega = numpy.fft.fftshift(numpy.fft.fftfreq(len(desired), d=1.0)) * 2.0*numpy.pi;
	num_nodes = len(omega);

	#===Plot Desired Frequency Response===#
	plt.figure(42);
	desired_mag = abs(desired);
	desired_log_mags = 20.0*numpy.log10(desired_mag**2.0);
	plt.plot(omega/(2.0*numpy.pi),desired_log_mags);
	plt.axhline(y=-3.0,linewidth=1,color='black');
	minn = min(desired_log_mags);
	plt.ylim([minn, 10.0]);
	plt.title("Desired Response");

	if (0):
		#===Make New Angle Response===#
		angles = numpy.angle(desired);
		angles = numpy.unwrap(angles);
		new_angles = numpy.zeros_like(angles);
		num_angles_between = 0;
		for rad_freq in omega:
			nyquist_freq = rad_freq/(2.0*numpy.pi);
			if ( nyquist_freq > -stopband and nyquist_freq < stopband ):  
				num_angles_between += 1;
		end = -float(2*filter_length+4);
		transition = numpy.linspace(0, end, num_angles_between);
		count = 0;
		for w, rad_freq in enumerate(omega):
			nyquist_freq = rad_freq/(2.0*numpy.pi);
			if (nyquist_freq < -stopband):
				new_angles[w] = 0.0;
			if (nyquist_freq > -stopband and nyquist_freq < stopband):
				new_angles[w] = transition[count];
				count += 1;
			if (nyquist_freq > stopband):
				new_angles[w] = end;
		angles = numpy.copy(new_angles);

		#===Update Desired Response===#
		desired = desired_mag * numpy.exp(1j * angles);	

	#===Plot Time Domain Response===#
	plt.figure(45);
	t = numpy.linspace(0, len(h_desired), len(h_desired));
	markerline, stemlines, baseline = plt.stem(t, h_desired)
	plt.setp(stemlines, 'color', 'red', 'linewidth', 3);
	plt.setp(baseline, 'color','black', 'linewidth', 2)	
	plt.title("Desired Time Domain Response");



	
if (not use_scipy):
	#===Create Frequency Nodes===#
	omega = numpy.pi/float(num_nodes) * numpy.linspace(0, num_nodes, num_nodes); 
	omega_pass = 0.45 * numpy.pi; #this works for 0.4 passband
	omega_stop = 0.5 * numpy.pi; #this works for 0.5 passband


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

#===Form Cost Vector===#
f = matrix(numpy.asmatrix(numpy.r_[numpy.zeros(filter_length), 1.0]).T);

#===Form Desired Response===#
if (not use_scipy):
	w = 0; 
	amplitude = 10e-8*numpy.ones(num_nodes);
	while(omega[w] <= omega_pass):
		amplitude[w] = 1.0;
		w = w + 1;
	phase = numpy.exp(1j * numpy.linspace(0, 2.0*numpy.pi, num_nodes));
	desired = amplitude * phase;

#===Form bs vector===#
bs = [];
for h in desired:
	b1 = numpy.matrix([-1.0*h.real, -1.0*h.imag]).T;
	bs.append(b1);

#===Form As matrix===#
As = [];
for omega_k in omega:
	vector = numpy.matrix([numpy.cos(i * omega_k) for i in range((filter_length))]);
	vector = numpy.c_[vector, 0];
	vector2 = numpy.matrix([-numpy.sin(i * omega_k) for i in range((filter_length))]);
	vector2 = numpy.c_[vector2, 0];
	A1 = numpy.r_[vector, vector2];
	As.append(A1);

#===Form cs vectors===#
cs = [];
ds = [];
for c in range(num_nodes):
	c1 = numpy.asmatrix(numpy.r_[numpy.zeros(filter_length), 1.0]).T;
	cs.append(c1);
	ds.append(numpy.matrix([0.0]).T);


#==================================================================================================#
#=========================================CONVERSION===============================================#
#==================================================================================================#

#===Form Block Matrices===#
Gs = [];
for a in range(len(As)):
	G1 = (numpy.r_[-cs[a].T, -As[a]]);
 	Gs.append(matrix(G1));
G = (lambda x, y: x+y, Gs)[1];

#===Form Block Vectors===#
hs = [];
for b in range(len(bs)):
	h1 = (numpy.r_[ds[b], bs[b]]);
 	hs.append(matrix(h1));
h = (lambda x, y: x+y, hs)[1];

#===Solve===#
sol = solvers.socp(f, Gq=G, hq=h);
print sol['status'];

#===Extract Filter===#
h_found = numpy.zeros(filter_length);
h_found = numpy.asarray(numpy.asmatrix(sol['x'][0:filter_length]))[:,0];
#h_found *= -1.0;

if (0):
	#===Ensure Symmetry===#
	for i in range(len(h_found)/2):
		temp1 = h_found[i];
		temp2 = h_found[len(h_found)-i-1];
		avg = 0.5*(temp1 + temp2);
		h_found[i] = h_found[len(h_found)-i-1] = avg;

#===Ensure Unity Gain===#
#h_found /= sum(h_found);

#===Plot Found Time Domain Response===#
plt.figure(4);
t = numpy.linspace(0, len(h_found), len(h_found));
markerline, stemlines, baseline = plt.stem(t, h_found)
plt.setp(stemlines, 'color', 'red', 'linewidth', 3);
plt.setp(baseline, 'color','black', 'linewidth', 2)	
plt.title("Found Time Domain Response");

#===Plot Found Frequency Domain Response===#
plt.figure(3);
found_fft = numpy.fft.fftshift(numpy.fft.fft(h_found, n=num_nodes));
found_freqs = numpy.fft.fftshift(numpy.fft.fftfreq(len(found_fft)));
mags = abs(found_fft);
log_mags = 20.0*numpy.log10(mags**2.0);
plt.plot(found_freqs,log_mags);
plt.axhline(y=-3.0,linewidth=1,color='black');
plt.ylim([minn, 10.0]);
plt.title("Solution Response");
plt.show();

