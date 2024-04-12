#This file is used for ML inference with trained model
import numpy as np
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import Data
#Load model
model_name="model1.keras"
model = keras.models.load_model('save/'+model_name)
model.summary()
#Load data
patch= Data.get_patch("patch/patch530.csv")[0]
inputs = tf.reshape(tf.convert_to_tensor(patch,dtype=tf.float32),(-1,32,32))
output= Data.get_Spara("data/data530.csv")
#Predicition

y=model.predict(inputs)


#comparaison 
plt.figure()
plt.plot(y[0][:101],'--',label="S11 predicted") #S11
plt.plot(output[0],label="S11 simulated")
plt.legend()
plt.xlabel("Frequency (100 MHz step)")
plt.ylabel("S parameters (mag)")
plt.ylim([-0.1,1])

plt.plot(y[0][101:202],'--',label="S21 predicted") #S21
plt.plot(output[1],label="S21 simulated")
plt.xlabel("Frequency (100 MHz step)")
plt.ylabel("S parameters (mag)")
plt.ylim([-0.1,1])
plt.legend()


plt.plot(y[0][303:404],'--',label="S31 predicted") #S31
plt.plot(output[3],label="S31 simulated")
plt.xlabel("Frequency (100 MHz step)")
plt.ylabel("S parameters (mag)")
plt.ylim([-0.1,1])
plt.legend()

plt.figure()
plt.plot(y[0][505:606],'--*',label="S41 predicted") #S23
plt.plot(output[5],label="S41 simulated")
plt.xlabel("Frequency (100 MHz step)")
plt.ylabel("S parameters (mag)")
plt.ylim([-0.1,1])
plt.legend()
plt.show()
