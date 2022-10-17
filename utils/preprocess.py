import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings("ignore")
import biosppy.signals.tools as st
import numpy as np
import os
import wfdb
from biosppy.signals.ecg import correct_rpeaks, hamilton_segmenter
from scipy.signal import medfilt
import neurokit2 as nk

# PhysioNet Apnea-ECG dataset
# /home/ubuntu/tu.na/sleepapnea/data/physionet.org/SA-2.0.0
# url: https://physionet.org/physiobank/database/apnea-ecg/
base_dir = "/home/ubuntu/tu.na/sleepapnea/data/physionet.org/SA-2.0.0/apnea-ecg/1.0.0/"
base_save = "/home/ubuntu/tu.na/sleepapnea/"

fs = 100
sample = fs * 60  # 1 min's sample points

before = 2  # forward interval (min)
after = 2  # backward interval (min)
hr_min = 20
hr_max = 300

num_worker = 55  # Setting according to the number of CPU cores

def find_S_point(ecg, R_peaks):
	num_peak=R_peaks.shape[0]
	S_point=list()
	for index in range(num_peak):
		i=R_peaks[index]
		cnt=i
		if cnt+1>=ecg.shape[0]:
			break
		while ecg[cnt]>ecg[cnt+1]:
			cnt+=1
			if cnt>=ecg.shape[0]:
				break
		S_point.append(cnt)
	return np.asarray(S_point)

def find_Q_point(ecg, R_peaks):
	num_peak=R_peaks.shape[0]
	Q_point=list()
	for index in range(num_peak):
		i=R_peaks[index]
		cnt=i
		if cnt-1<0:
			break
		while ecg[cnt]>ecg[cnt-1]:
			cnt-=1
			if cnt<0:
				break
		Q_point.append(cnt)
	return np.asarray(Q_point)

def worker(name, labels):
    print("processing %s!" % name)
    X = []
    y = []
    groups = []
    signals = wfdb.rdrecord(os.path.join(base_dir, name), channels=[0]).p_signal[:, 0] # Read recording
    for j in range(len(labels)):
        if j < before or \
                (j + 1 + after) > len(signals) / float(sample):
            continue
        signal = signals[int((j - before) * sample):int((j + 1 + after) * sample)]
        signal, _, _ = st.filter_signal(signal, ftype='FIR', band='bandpass', order=int(0.3 * fs),
                                        frequency=[3, 45], sampling_rate=fs) # Filtering the ecg signal to remove noise
        # Find R peaks
        rpeaks, = hamilton_segmenter(signal, sampling_rate=fs) # Extract R-peaks
        rpeaks, = correct_rpeaks(signal, rpeaks=rpeaks, sampling_rate=fs, tol=0.1)
        # Find S peaks
        if len(rpeaks) / (1 + after + before) < 40 or \
                len(rpeaks) / (1 + after + before) > 200:  # Remove abnormal R peaks signal
            continue
        
        speaks= find_S_point(signal, rpeaks)
        qpeaks=find_Q_point(signal, rpeaks)


        rri_tm, rri_signal = rpeaks[1:] / float(fs), np.diff(rpeaks) / float(fs)
        rri_signal = medfilt(rri_signal, kernel_size=3)

        # print(rpeaks)
        r_ampl_tm, r_ampl_siganl = rpeaks / float(fs), signal[rpeaks]

        s_ampl_tm  = speaks / float(fs)
        s_ampl_siganl  = signal[speaks]
        
        q_ampl_tm  = qpeaks / float(fs)
        q_ampl_siganl  = signal[qpeaks]


        hr = 60 / rri_signal
        
        # Remove physiologically impossible HR signal
        if np.all(np.logical_and(hr >= hr_min, hr <= hr_max)):
            # Save extracted signal
            
            X.append([(rri_tm, rri_signal), (r_ampl_tm, r_ampl_siganl),(s_ampl_tm, s_ampl_siganl),(q_ampl_tm, q_ampl_siganl)])
            y.append(0. if labels[j] == 'N' else 1.)
            groups.append(name)

    print("over %s!" % name)
    return X, y, groups


if __name__ == "__main__":
    apnea_ecg = {}

	# training dataset
    names = [
        "a01", "a02", "a03", "a04", "a05", "a06", "a07", "a08", "a09", "a10",
        "a11", "a12", "a13", "a14", "a15", "a16", "a17", "a18", "a19", "a20",
        "b01", "b02", "b03", "b04", "b05",
        "c01", "c02", "c03", "c04", "c05", "c06", "c07", "c08", "c09", "c10"
    ]
    
    # names = [
    #     "b04"
    # ]

    o_train = []
    y_train = []
    groups_train = []
    print('Training...')
    with ProcessPoolExecutor(max_workers=num_worker) as executor: # Speed up with parallel processing
        task_list = []
        for i in range(len(names)):
            labels = wfdb.rdann(os.path.join(base_dir, names[i]), extension="apn").symbol
            task_list.append(executor.submit(worker, names[i], labels))

        for task in as_completed(task_list):
            # print(task.result())
            X, y, groups = task.result()
            o_train.extend(X)
            y_train.extend(y)
            groups_train.extend(groups)

    print()

    answers = {}
    with open(os.path.join(base_dir, "event-2-answers"), "r") as f:
        for answer in f.read().split("\n\n"):
            answers[answer[:3]] = list("".join(answer.split()[2::2]))

	# testing dataset
    names = [
        "x01", "x02", "x03", "x04", "x05", "x06", "x07", "x08", "x09", "x10",
        "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x18", "x19", "x20",
        "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "x29", "x30",
        "x31", "x32", "x33", "x34", "x35"
    ]
    
    # names = [
    #     "x20"
    # ]

    o_test = []
    y_test = []
    groups_test = []
    print("Testing...")
    with ProcessPoolExecutor(max_workers=num_worker) as executor: # Speed up with parallel processing
        task_list = []
        for i in range(len(names)):
            labels = answers[names[i]]
            task_list.append(executor.submit(worker, names[i], labels))

        for task in as_completed(task_list):
            X, y, groups = task.result()
            o_test.extend(X)
            y_test.extend(y)
            groups_test.extend(groups)

	# Save preprocessing result
    apnea_ecg = dict(o_train=o_train, y_train=y_train, groups_train=groups_train,o_test=o_test, y_test=y_test, groups_test=groups_test)
    with open(os.path.join(base_save, "apnea-ecg_3-all.pkl"), "wb") as f:
        pickle.dump(apnea_ecg, f, protocol=2)

    print("\nok!")