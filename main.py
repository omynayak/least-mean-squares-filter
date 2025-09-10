import numpy as np
import matplotlib.pyplot as plt
import csv

def lms_filter(accel_signal, ppg_signal, mu=1e-5, filter_order=32):
    n_samples = len(ppg_signal)
    weights = np.zeros(filter_order)
    motion_approx = np.zeros(n_samples)
    cleaned_ppg = np.zeros(n_samples)

    for i in range(filter_order, n_samples):
        x = accel_signal[i-filter_order:i][::-1]   
        y = np.dot(weights, x)                     
        error = ppg_signal[i] - y                  
        weights = weights + 2 * mu * error * x     
        motion_approx[i] = y
        cleaned_ppg[i] = error                     

    return cleaned_ppg, motion_approx, weights


timeStamps = []
ppgData = []
with open("PPG.csv") as file:
    data = csv.reader(file)
    next(data)
    for row in data:
        timeStamps.append(int(row[1]))
        ppgData.append(float(row[2]))

timeStamps = np.array(timeStamps)
ppgData = np.array(ppgData)


accData = []
with open("ACC.csv", mode="r") as file:
    data = csv.reader(file)
    next(data)
    for row in data:
        tempX = float(row[2])
        tempY = float(row[3])
        tempZ = float(row[4])
        mag = (tempX**2 + tempY**2 + tempZ**2) ** 0.5
        accData.append(mag)

accData = np.array(accData)


cleaned_signal, motion_approx, weights = lms_filter(accData, ppgData, mu=1e-5, filter_order=32)


plt.figure(figsize=(12,5))
plt.subplot(1,3,1)
plt.plot(timeStamps, ppgData, label="Noisy PPG")
plt.plot(timeStamps, cleaned_signal, label="Cleaned PPG")
plt.legend()
plt.title("PPG before vs after LMS")
plt.grid()


plt.subplot(1,3,2)
plt.plot(timeStamps, accData, label= "Accelerometer signals")
plt.legend()
plt.title("Accelerometer data plot")
plt.grid()

plt.subplot(1,3,3)
plt.plot(timeStamps, motion_approx, label="Motion estimate")
plt.legend()
plt.title("Motion Artifact Estimated from ACC")
plt.grid()

plt.tight_layout()
plt.show()
