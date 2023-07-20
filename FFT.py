import numpy as np
import matplotlib.pyplot as plt

# sampling frequency
fs = 500
dura = 2
N = fs*dura
time = np.arange(0, dura, 1/fs)
dt = 1/fs

# parameters
a1, f1, p1 = 3, 30, 0.6
a2, f2, p2 = 2, 45, -0.8
a3, f3, p3 = 1, 70, 2

# signals
s1 = a1 * np.cos(2*np.pi*f1*time + p1)
s2 = a2 * np.cos(2*np.pi*f2*time + p2)
s3 = a3 * np.cos(2*np.pi*f3*time + p3)
noise = 2 * np.random.randn(len(time))
s = s1 + s2 + s3
s_noise = s1 + s2 + s3 + noise

# plot time domain
plt.figure(1)
plt.subplot(2,1,1)
plt.plot(time, s_noise, label='s noise')
plt.plot(time, s, label='s')
plt.legend()
plt.title('FFT')
plt.xlabel('sec')
plt.ylabel('amp')

# fft
f_half = np.arange(0, int(N/2))*fs/N  # rescale
# f_half = 1/(dt*N) * np.arange(int(N/2))
S = np.fft.fft(s_noise, N)
S_half = S[0:int(N/2)]
S_mag = 2*np.abs(S_half)/N  # rescale
# PSD = S * np.conj(S) / N  # power spectrum density

# plot freq domain
plt.subplot(2,1,2)
plt.plot(f_half, S_mag, label='S noise')
plt.legend()
plt.xlabel('freq')
plt.ylabel('amp')

# calculate phases
# ph1 = np.angle(S_half[f1*dura])
# ph2 = np.angle(S_half[f2*dura])
# ph3 = np.angle(S_half[f3*dura])
# print(ph1)
# print(ph2)
# print(ph3)

# zeros out noise under threshold
threshold = 0.5
# indices = S_mag > threshold  # create a mask of 0 or 1
# S_mag_clean = S_mag * indices
S_mag_clean = np.where(S_mag < threshold, 0, S_mag)

# plot freq domain
plt.figure(2)
plt.subplot(2,1,1)
plt.plot(f_half, S_mag, label='S noise')
plt.plot(f_half, S_mag_clean, label='S')
plt.plot(f_half, threshold*np.ones(len(f_half)), label='threshold')
plt.legend()
plt.title('Reconstruct signal')
plt.xlabel('freq')
plt.ylabel('amp')

# extract parameters
save_a = []
save_f = []
save_p = []
for index, value in enumerate(S_mag_clean):
    if value != 0:
        save_a.append(value)
        save_f.append(index*(fs/N))  # rescale

# reconstruct signal
recon = np.zeros(np.size(time))
for i in range(0, len(save_a)):
    phi = np.angle(S_half[int(save_f[i] * dura)])
    save_p.append(phi)
    recon = recon + save_a[i] * np.cos(2*np.pi*save_f[i]*time + phi)
print(f'amp: {save_a}\nfre: {save_f}\nphi: {save_p}')

# plot time domain
plt.subplot(2,1,2)
plt.plot(time, s, label='s')
plt.plot(time, recon, '--', label='recon')
plt.legend()
plt.xlabel('sec')
plt.ylabel('amp')

# IFFT with noise
# s2 = np.fft.ifft(S, N)

# # plot time domain
# plt.figure(3)
# plt.subplot(2,1,1)
# plt.plot(time, s_noise, label='s noise')
# plt.plot(time, np.real(s2), '--', label='ifft')
# plt.legend()
# plt.title('IFFT')
# plt.xlabel('sec')
# plt.ylabel('amp')

# PSD
threshold2 = 100
PSD = S*np.conj(S)/N
PSD_clean = np.where(PSD < threshold2, 0, S)

# plot freq domain
plt.figure(3)
plt.subplot(2,1,1)
plt.plot(f_half, np.real(PSD[0:int(N/2)]), label='PSD')
plt.plot(f_half, np.real(PSD_clean[0:int(N/2)]), label='clean PSD')
plt.plot(f_half, threshold2*np.ones(len(f_half)), label='threshold')
plt.title('IFFT')
plt.legend()
plt.xlabel('freq')
plt.ylabel('amp')

# IFFT
# indices2 = S > threshold
# temp = S * indices2
s_clean = np.fft.ifft(PSD_clean, N)

# plot time domain
plt.subplot(2,1,2)
plt.plot(time, s, label='s')
plt.plot(time, np.real(s_clean), '--', label='ifft')
plt.legend()
plt.xlabel('sec')
plt.ylabel('amp')

# DFT cos sin coefficients
a0 = S[0]/N
an = 2*np.real(S[1:int(N/2)+1])/N
bn = 2*np.imag(S[1:int(N/2)+1])/N
amp = np.sqrt(an*an + bn*bn)
pha = np.arctan2(bn, an)
# plt.figure(4)
# plt.plot(f_half, amp)

plt.show()

# https://www.youtube.com/watch?v=XEbV7WfoOSE
