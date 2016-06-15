import numpy as np
from scipy.io import wavfile

# found the solution to conform to MATLAB style wavwrite at
# https://github.com/sam81/wavpy
def wavwrite(data, fs, nbits, fname):
    if nbits not in [16, 32]:
        raise ValueError('Only 16bit and 32bit formats are supported!')

    formats = {16: np.int16, 32: np.int32}
    # by experiment, round the float to int using np.rint
    # makes it much closer to the values produced by MATLAB wavwrite(),
    # but still can't eliminate some rounding errors (+/- 1). I wonder
    # how MATLAB wavwrite() converts from float to int
    data = np.rint(data * (2**(nbits - 1))).astype(formats[nbits])
    wavfile.write(fname, fs, data)


data_base = 'data/q4/'

mix = np.loadtxt(data_base + 'mix.dat')  # load mixed sources
m, n = mix.shape  # we know n=5
Fs = 11025  #sampling frequency being used

# normalize each of five signals
normalizedMix = 0.99 * mix / np.max(np.abs(mix), axis=0)

# write mixed wav files
wavwrite(normalizedMix[:, 0], Fs, 16, data_base + 'mix1.wav')
wavwrite(normalizedMix[:, 1], Fs, 16, data_base + 'mix2.wav')
wavwrite(normalizedMix[:, 2], Fs, 16, data_base + 'mix3.wav')
wavwrite(normalizedMix[:, 3], Fs, 16, data_base + 'mix4.wav')
wavwrite(normalizedMix[:, 4], Fs, 16, data_base + 'mix5.wav')

W = np.eye(n)  # initialize unmixing matrix

learning_rates = [0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.02, 0.02, 0.01, 0.01,
                  0.005, 0.005, 0.002, 0.002, 0.001, 0.001]

for a in learning_rates:
    # Your code

# After finding W, use it to unmix the sources.  Place the unmixed sources
# in the matrix S (one source per column).  (Your code.)

# rescale each column to have maximum absolute value 1
S = 0.99 * S / np.max(np.abs(S), axis=0)

# now have a listen --- You should have the following five samples:
# * Godfather
# * Southpark
# * Beethoven 5th
# * Austin Powers
# * Matrix (the movie, not the linear algebra construct :-)

# write back to wav files
wavwrite(S[:, 0], Fs, 16, data_base + 'unmix1.wav')
wavwrite(S[:, 1], Fs, 16, data_base + 'unmix2.wav')
wavwrite(S[:, 2], Fs, 16, data_base + 'unmix3.wav')
wavwrite(S[:, 3], Fs, 16, data_base + 'unmix4.wav')
wavwrite(S[:, 4], Fs, 16, data_base + 'unmix5.wav')
