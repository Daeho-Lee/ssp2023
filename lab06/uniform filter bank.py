import librosa
import numpy as np
import matplotlib.pyplot as plt


wav_path = r"C:\Users\IT2-000\PycharmProjects\lab02\lab04\lab04_mix0db.wav"      # mix-10db, mix0db, mix10db, mix20db
mix_wav, sr = librosa.load(wav_path, sr=16000)    # x = mix signal

Ts = 0.01   # 10 ms shift size
Tf = 0.02   # 20 ms frame size
Ns = int(sr*Ts)    # shift number of samples
Nf = int(sr*Tf)    # frame number of samples
NFFT = int(2**(np.ceil(np.log2(Nf))))
hNo = NFFT//2+1
filter_tab = 31
wav_nframe = int((len(mix_wav)-Nf)/Ns)
output = []

for ii in np.arange(wav_nframe):

    x = mix_wav[ii*Ns:ii*Ns+Ns]
    X = np.fft.fft(x, NFFT)
    magX = (X*np.conj(X)).real
    magX = magX[:int(len(X)/2+1)]
    output.append(magX)

output = np.array(output)
filter_bank = np.linspace(0,1)np.linspace(0,1, int(filter_tab/2+1))