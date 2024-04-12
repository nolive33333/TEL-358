#This file is used for automaticaly simulated parten which is randomly generated. It requires to_HFSS.py for build and manage HFSS software. 


#/!\/!\/!\/!\ A crach occur after many simulations. It seems due to PyAEDT provided by Ansys but for now I didn't succed to overcome this issue. So every 5 hours we need to restart again 

import numpy as np 
import Data
import to_HFSS
import pyaedt
import shutil
import os
import matplotlib.pyplot as plt 
import random
from datetime import datetime


f1=int(5e9)
f2=int(15e9)
project_HFSS=0
fc1=7e9
fc2=11e9
f=np.linspace(f1,f2,101)


n=32 #number of patch 32x32
k=1171   #number of new data             

tr=0
while(tr<5):
    try:   #attempt to overcome the crash but doesn't work.... 
        hfss, patch, setup=to_HFSS.draw_strucutre(30,1.1,f1,f2)
        while(k<10000):
            #randomly generate patern, many possible strategies
            output=[1]*32+[0]*(1024-2*32)+[1]*32
            for i in range(32,1024-32):
                output[i]=random.randint(0,1)

            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print(current_time," Run num:",k)        
            to_HFSS.mise_a_jour_cond(hfss,output,patch,k)
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print(current_time," Simulation go")    
            to_HFSS.run_and_get_data(hfss,setup,k)
            k+=1
    except none_dealloc:
        print("error, another test will be try")
    tr+=1
