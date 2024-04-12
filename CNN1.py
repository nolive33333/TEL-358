#This file contain the CNN model and the training part associated 

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import datasets, layers, models, regularizers
import matplotlib.pyplot as plt
from datetime import datetime
import Data # for data pre processing 
import math
#Some optional function/class declaration 

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

class printlearningrate(tf.keras.callbacks.Callback): #Use this for plot loss and learning rate during training 
    def __init__(self):
        self.lr=[]
        self.loss=[]
        self.loss_val=[]
    def on_epoch_end(self, epoch, loss):
        optimizer = self.model.optimizer
        self.lr.append(K.eval(optimizer.learning_rate))
        self.loss.append(float(loss['loss']))
        self.loss_val.append(float(loss['val_loss']))
        Epoch_count = epoch + 1
        # plt.subplot(211)
        plt.plot(self.loss)
        plt.plot(self.loss_val)
        plt.yscale('log')
        # plt.subplot(212)
        # plt.plot(self.lr)
        plt.draw()
        plt.pause(0.01)
        plt.clf()

class CosineScheduler:
    def __init__(self, max_update, base_lr=0.01, final_lr=0,
               warmup_steps=0, warmup_begin_lr=0):
        self.base_lr_orig = base_lr
        self.max_update = max_update
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.warmup_begin_lr = warmup_begin_lr
        self.max_steps = self.max_update - self.warmup_steps

    def get_warmup_lr(self, epoch):
        increase = (self.base_lr_orig - self.warmup_begin_lr) \
                       * float(epoch) / float(self.warmup_steps)
        return self.warmup_begin_lr + increase

    def __call__(self, epoch):
        if epoch < self.warmup_steps:
            return self.get_warmup_lr(epoch)
        if epoch <= self.max_update:
            self.base_lr = self.final_lr + (
                self.base_lr_orig - self.final_lr) * (1 + math.cos(
                math.pi * (epoch - self.warmup_steps) / self.max_steps)) / 2
        return self.base_lr
#############################
#                           #
#     model definition      #
#                           #
#############################
cdrop=0.5 #coef for droupout layer 

cl2=0           # l2 penalty for karnel
cl2bias=0       # l2 penalty for bias
cl2activity=0   # l2 penalty for activation

model = models.Sequential()
model.add(layers.Conv2D(64, (9, 9), input_shape=(32, 32, 1),activation = 'relu',padding="same", kernel_regularizer=regularizers.L2(cl2), bias_regularizer=regularizers.L2(cl2bias), activity_regularizer=regularizers.L2(cl2activity)))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.SpatialDropout2D(cdrop))
model.add(layers.Conv2D(128, (5, 5),activation = 'relu', padding="same",kernel_regularizer=regularizers.L2(cl2), bias_regularizer=regularizers.L2(cl2bias), activity_regularizer=regularizers.L2(cl2activity)))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.SpatialDropout2D(cdrop))
model.add(layers.Conv2D(512, (3, 3),activation = 'relu', padding="same",kernel_regularizer=regularizers.L2(cl2), bias_regularizer=regularizers.L2(cl2bias), activity_regularizer=regularizers.L2(cl2activity)))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.SpatialDropout2D(cdrop))
model.add(layers.Conv2D(1010, (2, 2),activation = 'relu', padding="same",kernel_regularizer=regularizers.L2(cl2), bias_regularizer=regularizers.L2(cl2bias), activity_regularizer=regularizers.L2(cl2activity)))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((4, 4)))

model.add(layers.SpatialDropout2D(cdrop))
model.add(layers.Conv2D(1010, (1, 1),activation = 'linear', padding="same",kernel_regularizer=regularizers.L2(cl2), bias_regularizer=regularizers.L2(cl2bias), activity_regularizer=regularizers.L2(cl2activity)))

#OUTPUT
model.add(layers.Flatten())
#model.add(layers.Dense(1010,activation='linear'))  #fully connected layer
model.summary()
shape=model.output_shape

#############################
#                           #
#     Gestion Dataset       #
#                           #
#############################



images, labels,name = Data.get_data(True) #recover data with augmted option = True (allow to multiply by 8 the number of data)
images_s,labels_s,name_s=Data.shuffle_data(images,labels,name) #apply shuffle 
DATASET_SIZE=len(images)
train_size=int(0.8*DATASET_SIZE)
val_size=int(0.15*DATASET_SIZE)
test_size=int(0.05*DATASET_SIZE)


#Tensor convertion
full_images = tf.convert_to_tensor(images_s, dtype=tf.float32)
full_labels = tf.convert_to_tensor(labels_s, dtype=tf.float32)


train_images = full_images[0:train_size]
train_labels = full_labels[0:train_size]

val_images = full_images[train_size:train_size+val_size]
val_labels = full_labels[train_size:train_size+val_size]

test_images = full_images[train_size+val_size:]
test_labels = full_labels[train_size+val_size:]
test_name   = name_s[train_size+val_size:]


print("Dataset size = ",DATASET_SIZE)
print("Train dataset size = ", train_size)

#############################
#                           #
#     Start training        #
#                           #
#############################

#kers call backs
printlr = printlearningrate() 
updatelr = tf.keras.callbacks.LearningRateScheduler(CosineScheduler(30, warmup_steps=5, base_lr=0.002, final_lr=0.001,warmup_begin_lr=0.002)) #Not use here, if want to use it, add in callbacks when training 
#Use for tensorflow logs 
log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.compile(optimizer='Adam',
              loss=tf.keras.losses.MeanSquaredError(),
                metrics=[tf.keras.metrics.RootMeanSquaredError()])


print(datetime.now().strftime("%H:%M:%S")," Start training:")

history = model.fit(train_images, train_labels,batch_size=16, epochs=200,verbose=1, 
                    validation_data=(val_images, val_labels),callbacks=[printlr,tensorboard_callback])

print(datetime.now().strftime("%H:%M:%S")," Done:")

plt.plot(printlr.loss)
plt.plot(printlr.loss_val)
plt.show()
#Evaution of the model and try prediction
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print("model accuracy: ",test_acc)

y=model.predict(test_images)
plt.plot(y[2][:101],label='y2')
plt.plot(y[1][:101],label='y1')



plt.plot(test_labels[1][:101],label=test_name[1]+'label1')
plt.plot(test_labels[2][:101],label=test_name[2]+'label2')

plt.legend()
plt.show()

#Save data
Data.save([printlr.loss,printlr.loss_val,printlr.lr],"save/test.csv")
model.save('save/model1.keras')  # The file needs to end with the .keras extension