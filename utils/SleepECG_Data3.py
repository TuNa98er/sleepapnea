from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import pickle
from scipy.interpolate import splev, splrep


base_dir = "./apnea-ecg.pkl"
ir = 3 # interpolate interval
before = 2
after = 2
scaler = lambda arr: (arr - np.min(arr)) / (np.max(arr) - np.min(arr))


class Sleep_Agnea_DB(Dataset):
    def __init__(self, data, label):
        self.data=data
        self.label=label
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data= self.data[idx].transpose()
        label=int(self.label[idx])
        return data, label


def load_data():
    tm = np.arange(0, (before + 1 + after) * 60, step=1 / float(ir))

    with open(os.path.join("./apnea-ecg.pkl"), 'rb') as f: # read preprocessing result
        apnea_ecg = pickle.load(f)

    x_train = []
    o_train, y_train = apnea_ecg["o_train"], apnea_ecg["y_train"]
    groups_train = apnea_ecg["groups_train"]
    for i in range(len(o_train)):
        # (rri_tm, rri_signal), (r_ampl_tm, r_ampl_siganl),(s_ampl_tm, s_ampl_siganl)  = o_train[i]
        (rri_tm, rri_signal), (r_ampl_tm, r_ampl_siganl) = o_train[i]
		# Curve interpolation
        rri_interp_signal = splev(tm, splrep(rri_tm, scaler(rri_signal), k=3), ext=1) 
        r_ampl_interp_signal = splev(tm, splrep(r_ampl_tm, scaler(r_ampl_siganl), k=3), ext=1)
        # s_ampl_interp_signal = splev(tm, splrep(s_ampl_tm, scaler(s_ampl_siganl), k=3), ext=1)
        
        x_train.append([rri_interp_signal, r_ampl_interp_signal])
        # print(x_train)
    x_train = np.array(x_train, dtype="float32").transpose((0, 2, 1)) # convert to numpy format
    y_train = np.array(y_train, dtype="float32")

    x_test = []
    o_test, y_test = apnea_ecg["o_test"], apnea_ecg["y_test"]
    groups_test = apnea_ecg["groups_test"]
    for i in range(len(o_test)):
        (rri_tm, rri_signal), (r_ampl_tm, r_ampl_siganl) = o_test[i]
		# Curve interpolation
        rri_interp_signal = splev(tm, splrep(rri_tm, scaler(rri_signal), k=3), ext=1)
        r_ampl_interp_signal = splev(tm, splrep(r_ampl_tm, scaler(r_ampl_siganl), k=3), ext=1)

        # s_ampl_interp_signal = splev(tm, splrep(s_ampl_tm, scaler(s_ampl_siganl), k=3), ext=1)

        x_test.append([rri_interp_signal, r_ampl_interp_signal])
        # print(x_test)
    # print()   
    x_test = np.array(x_test, dtype="float32").transpose((0, 2, 1))
    y_test = np.array(y_test, dtype="float32")

    return x_train, y_train, groups_train, x_test, y_test, groups_test