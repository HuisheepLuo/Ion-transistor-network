from numba.experimental import jitclass
from numba import int32, float64
from scipy.optimize import curve_fit
import numpy as np

sigma = 0
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
        noise = 1 - np.random.rand(self.num+1,) * sigma

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
            ID[i+1] = self.VS[i] * G_array[i+1] * noise[i]
        self.G_array = G_array * noise
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

