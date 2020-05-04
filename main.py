import numpy as np
import random
import math
import matplotlib.pyplot as plt
import time


AMPLITUDE_MAX = 1
FULL_CIRCLE = 2 * math.pi

HARMONICS = 14
TICKS = 64
FREQUENCY = 1700


def random_signal(harmonics, ticks, freq):
    generated_signal = np.zeros(ticks)
    #     start = time.time()
    for i in range(harmonics):
        fi = FULL_CIRCLE * random.random()
        amplitude = AMPLITUDE_MAX * random.random()
        w = freq - i * freq / harmonics

        x = amplitude * np.sin(np.arange(0, ticks, 1) * w + fi)
        generated_signal += x
    return generated_signal


def correlation_tau(arr, exp_val):
    start = time.time()
    arr = np.array(arr)
    Rxx = np.zeros(len(arr))
    for tau in range(len(arr)):
        Rxx[tau] = np.sum((arr[:len(arr) - tau] - exp_val) * (arr[tau: len(arr)] - exp_val)) / (len(arr) - 1)
    return Rxx, time.time() - start


def correlation_tau_xy(arr1, arr2, exp_val1, exp_val2):
    start = time.time()
    arr1 = np.array(arr1)
    arr2 = np.array(arr2)
    Rxy = np.zeros(len(arr1))
    for tau in range(len(arr1)):
        Rxy[tau] = np.sum((arr1[:len(arr1) - tau] - exp_val1) * (arr2[tau: len(arr2)] - exp_val2)) / (len(arr1) - 1)
    return Rxy, time.time() - start


if __name__ == '__main__':
    random.seed(10)   # to make plots reproducible and the text positions configurable
# lab 1.1
    sig1 = random_signal(HARMONICS, TICKS, FREQUENCY)
    mean1 = np.mean(sig1)
    var1 = np.var(sig1)

    sig2 = random_signal(HARMONICS, TICKS, FREQUENCY)
    mean2 = np.mean(sig2)
    var2 = np.var(sig2)

    max_val = max(np.max(sig1), np.max(sig2))
    x_line = list(range(len(sig1)))

# plotting lab 1.1
    plt.subplot(311)
    p1 = plt.plot(x_line, sig1, label='Sig1')[0]
    plt.text(0, max_val + 0.5, 'Mean(1): %.3f' % mean1, fontsize=11)
    plt.text(0, max_val + 1, 'Variance(1): %.3f' % var1, fontsize=11)

    p2 = plt.plot(x_line, sig2, label='Sig2')[0]
    plt.text(32, max_val + 0.5, 'Variance(2) %.3f' % var2, fontsize=11)
    plt.text(32, max_val + 1, 'Mean(2): %.3f' % mean2, fontsize=11)

    plt.xlabel('t')
    plt.ylabel('f(t)')
    plt.legend(handles=[p1, p2])

# lab 1.2 (1)
    correlation1 = correlation_tau(sig1, mean1)[0]
    correlation2 = correlation_tau(sig2, mean2)[0]

    val1 = correlation1[0]
    val2 = correlation2[0]
    max_val = max(val1, val2)

# plotting lab 1.2 (1)
    plt.subplot(312)
    cp1 = plt.plot(x_line, correlation1, label='Rxx(sig1)')[0]
    plt.text(0, max_val + 0.2, 'Rxx1 (tau=0), %.4f' % val1)

    cp2 = plt.plot(x_line, correlation2, label='Ryy(sig2)')[0]
    plt.text(32, max_val + 0.2, 'Ryy2 (tau=0), %.4f' % val2)

    plt.xlabel('tau')
    plt.ylabel('R(tau)')
    plt.legend(handles=[cp1, cp2])

# lab 1.2 (2)
    correlation_xy = correlation_tau_xy(sig1, sig2, mean1, mean2)[0]

# plotting lab 1.2 (2)
    plt.subplot(313)
    cp3 = plt.plot(x_line, correlation_xy, label='Rxy')

    plt.xlabel('tau')
    plt.ylabel('R(tau)')
    plt.legend(handles=cp3)
# draw plots
    plt.show()
