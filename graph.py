# /!\ This file is only for display and verify data, it is not used for training or others things

import csv
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
import Data

def get_S21():
    with open("data/data1144.csv", 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        linecount = 0
        s11 = []
        s21 = []
        for row in reader:
            if linecount == 0:
                linecount += 1
            else:
                s11.append(float(row[5]))
                s21.append(float(row[6]))
    return s11, s21
S11,s21=get_S21()
S11dB=10*np.log10(S11)
s21dB=10*np.log10(s21)
plt.plot(S11dB)
plt.plot(s21dB)
plt.show()




file = Data.get_patch("patch/patch1.csv")
file1 = Data.get_patch("patch/patch803.csv")
file2 = Data.get_patch("patch/patch705.csv")
file3 = Data.get_patch("patch/patch302.csv")
file4 = Data.get_patch("patch/patch500.csv")
file5 = Data.get_patch("patch/patch613.csv")
plt.subplot(4,2,1)
plt.imshow(file4[0])
plt.xlabel("(a)")


plt.subplot(4,2,2)
plt.imshow(file4[1])
plt.xlabel("(b)")
plt.subplot(4,2,3)
plt.imshow(file4[2])
plt.xlabel("(c)")
plt.subplot(4,2,4)
plt.imshow(file4[3])
plt.xlabel("(d)")
plt.subplot(4,2,5)
plt.imshow(file4[4])
plt.xlabel("(e)")
plt.subplot(4,2,6)
plt.imshow(file4[5])
plt.xlabel("(f)")
plt.subplot(4,2,7)
plt.imshow(file4[6])
plt.xlabel("(g)")
plt.subplot(4,2,8)
plt.imshow(file4[7])
plt.xlabel("(h)")
plt.tight_layout(h_pad=0)
# plt.axis(False)
plt.show()