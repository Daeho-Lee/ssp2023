import librosa
import numpy as np

wav_path = r"C:\Users\IT2-000\PycharmProjects\lab02\lab03\kdigits3-1.wav"
wav, sr = librosa.load(wav_path, sr=16000)

T = len(wav)
Ts = 0.01   # 10 ms shift size
Tf = 0.02   # 20 ms frame size
Ns = int(sr*Ts)    # shift number of samples
Nf = int(sr*Tf)    # frame number of samples
K = int((T+Ns-1)/Ns)
M = 14

def lpc(y, m):
    R = [y.dot(y)]
    if R[0] == 0:
        return [1] + [0] * (m-2) + [-1]
    else:
        for i in range(1, m + 1):
            r = y[i:].dot(y[:-i])
            R.append(r)
        R = np.array(R)
        A = np.array([1, -R[1] / R[0]])
        E = R[0] + R[1] * A[1]
        for k in range(1, m):
            if (E == 0):
                E = 10e-17
            alpha = - A[:k+1].dot(R[k+1:0:-1]) / E
            A = np.hstack([A,0])
            A = A + alpha * A[::-1]
            E *= (1 - alpha**2)
        return A, E

def wavfilevocoder(x, M):
    for k in range(K):
        xk = x[(k*Ns):((k+1)*Ns)]
        A[:,k] = librosa.lpc(xk, M)

    return A, E

def preemphasis(x, xmem=0, alpha=0.98):
    xp = np.ndarray(x.shape)
    xp[0] = x[0] - alpha * xmem
    xp[1:] = x[1:] - alpha * xmem[0:-1]

    return xp

A,E = lpc(wav,M)