from numba.experimental import jitclass
from numba import int32, int64, float64, njit, jit
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import numpy as np

dt = 0.02
A = 2.54e-5
B = -3.63e-6
tau = 0.116
# G_max, G_min = 2.5713e-5, 2.1363e-5
G_max, G_min = 4e-5, 2.1363e-5

def func(x, A, B, t):
    return A + B * np.exp(-x/t)

def fitting_func(x_data, y_data):
    popt, pcov = curve_fit(func, x_data, y_data)
    A, B, tau = popt
    # plt.figure()
    # plt.plot(x_data, y_data)
    # plt.plot(x_data, func(x_data, A, B, tau))
    # print(popt)
    return popt


spec = [
    ('VG', float64[:]),
    ('VS', float64[:]),
    ('num', int32),
    ('dt', float64),
    ('ID', float64[:]),
    ('G_max', float64),
    ('G_min', float64),
    ('G_array', float64[:]),

]

@jitclass(spec)
class tft:
    def __init__(self, VG:list, VS:list, dt:float=dt):
        self.VG = VG
        self.VS = VS
        self.num = len(VG)
        self.dt = dt
        self.G_max, self.G_min = G_max, G_min

        self.ID = self.cur_drain()


    def cur_drain(self):
            # init state
        G_array = np.zeros(self.num+1,)
        G_array[0] = self.G_min
        ID = np.zeros(self.num+1,)
        ID[0] = self.VS[0] * G_array[0]
        spk_count = 0
        wav_count = 0

        for i in range(self.num):
            dt = self.dt
            if i > 0 and self.VG[i] == self.VG[i-1]:
                spk_count += 1
            elif self.VG[i] != self.VG[i-1]:
                spk_count = 0
                if self.VG[i] == 0:
                    wav_count += 1
                    # print(i, wav_count)
            t_now = spk_count * dt
            t_next = (spk_count + 1) * dt
            beta_p = 1 + 0.5 * np.exp(-wav_count / 0.8)
            beta_d = 1 - 0.5 * np.exp(-wav_count / 0.8)

            if self.VG[i] > 0:
                dG = B * beta_p * (np.exp(-t_next/tau) - np.exp(-t_now/tau))
            else:
                dG = - B * beta_d * (np.exp(-t_next/tau) - np.exp(-t_now/tau))
            G_array[i+1] = G_array[i] + dG
            if G_array[i+1] > self.G_max:
                G_array[i+1] = self.G_max
            elif G_array[i+1] < self.G_min:
                G_array[i+1] = self.G_min
            ID[i+1] = self.VS[i] * G_array[i+1]
        self.G_array = G_array
        return ID


def tft_rc(VG_group, VS):
    ID = np.zeros((len(VG_group)*4,), dtype=np.float64)
    count = 0
    for VG in VG_group:
        ID[count] = tft(VG[0:5], VS[0:5]).ID[-1]
        count += 1
        ID[count] = tft(VG[5:10], VS[5:10]).ID[-1]
        count += 1
        ID[count] = tft(VG[10:15], VS[10:15]).ID[-1]
        count += 1
        ID[count] = tft(VG[15:], VS[15:]).ID[-1]
        count += 1
    return ID


if __name__ == '__main__':
    plt.rcParams['font.size'] = 14
    plt.rcParams['font.sans-serif'] = ['Arial']
    num = 400
    wav = 10
    VG = np.zeros((num,), dtype=np.float64)
    for i in range(1, wav+1):
        VG[i*20:i*20+10] = 1.
    # VG = np.random.rand(num,)

    VS = np.ones((num,), dtype=np.float64)
    ## 1V
    # I_high = np.array([1.2, 2.7299, 2.8094, 2.8349, 2.8425, 2.8495])
    # I_low = np.array([1.2, 1.6856, 1.8138, 1.8794, 1.9316])

    ## 2V
    # I_high = np.array([1.2, 3.0744, 3.1668, 3.2008, 3.2220, 3.2415])
    # I_low = np.array([1.2, 2.8553, 2.9690, 3.0338, 3.0725])
    # A_high, B_high, tau_high = fitting_func(np.linspace(0, 5, 6), I_high)
    # A_low, B_low, tau_low = fitting_func(np.linspace(0, 4, 5), I_low)
    # plt.figure()
    # plt.plot(func(np.linspace(0, 5, 6), A_high, B_high, tau_high))
    # plt.plot(func(np.linspace(0, 4, 5), A_low, B_low, tau_low))

    tft0 = tft(VG, VS, 0.02)

    # plt.figure(figsize=(4.5,2))
    f, ax = plt.subplots(2, 1, figsize=(4.5,2), sharex=True)

    plt.xticks([])
    ax[0].set_yticks([])
    ax[1].set_yticks([])
    ax[0].set_ylim(-0.1,1.2)

    # plt.xlabel('Time')
    # plt.ylabel('Conductance')
    # ax[1].set_xlabel('Time')
    # ax[0].set_ylabel('Gate Voltage')
    # ax[1].set_ylabel('Conductance')
    ax[0].plot(VG)
    ax[1].plot(tft0.ID)
    plt.savefig('ltp_2.png')
    plt.show()