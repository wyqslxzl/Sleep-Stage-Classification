import tensorflow as tf
import numpy as np
import os
import h5py as h5
import time

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

train_D = 'F:/Work/EEG/SHHS/train/'
test_D = 'F:/Work/EEG/SHHS/test/'

out = 'F:/Work/model-SHHS/'
if os.path.exists(out) == 0:
    os.makedirs(out)

# Parameters
N_CLASSES = 5              # number of classes                            
NUM_ITER = 200             # number of iter time
ACTIVATION = tf.nn.relu    # activation function
LEARNING_RATE = 0.001

# Network
inputs = tf.keras.layers.Input(shape=(96,128) ,name='inputs')
LSTM = tf.keras.layers.LSTM(units=128, return_sequences=False)(inputs)
outputs = tf.keras.layers.Dense(units=5, activation='softmax', name='outputs')(LSTM)
LSTM = tf.keras.Model(inputs, outputs, name='LSTM')

# Cost
LSTM.compile(optimizer=tf.keras.optimizers.Adam(LEARNING_RATE), \
             loss='sparse_categorical_crossentropy', \
             metrics=['accuracy'])                 
LSTM.summary()

# Keep training until reach max iteration 
step = np.zeros(1).astype('int32')
step += 1
batch_x = np.zeros([4096,96,128],dtype='float32')
batch_y = np.zeros([4096],dtype='float32')

Dir_train= os.listdir( train_D )
Dir_test = os.listdir( test_D )

number = len(Dir_train) 
for i in range(0,number):
    Dir_train[i] = os.path.join( train_D , Dir_train[i] )

number = len(Dir_test) 
for i in range(0,number):
    Dir_test[i] = os.path.join( test_D , Dir_test[i] )

flag1 = 0;    
flag2 = 0;
before = 0;
while step <= NUM_ITER:
 
    # Train: Trainning Set 
    acc_num = np.zeros(1).astype('int32')
    Dir_train_T = np.array(Dir_train)
    number = len(Dir_train)
    num_block = 1
    times = round( number / num_block - 0.5 )
    start = time.time()
    for i in range(0,times):
        
        number = 0
        for j in range(0,num_block):      
            data = h5.File(  Dir_train_T[i*num_block+j] , 'r+' )
            tmp_x = np.array( data['im'] )
            tmp_y = np.array( data['la'] )
            data.close()
            length = tmp_x.shape[0]
            batch_x[number:number+length,:,:] = np.concatenate([tmp_x[:,:,:,0], tmp_x[:,:,:,1], tmp_x[:,:,:,2], tmp_x[:,:,:,3]], 2)
            batch_y[number:number+length] = np.argmax( tmp_y , 1 )
            number += length
                   
        history_train = LSTM.fit(batch_x[:number,:,:], batch_y[:number], batch_size=2048, epochs=1, verbose=0)
        acc_num += 1
         
        #time.sleep(0.5)
        
    if np.mod(step,5) == 0:
        tf.saved_model.save(LSTM, out+"%03d"%(step))            
    if np.mod(step,10) == 0:
        LEARNING_RATE /= 10
    start = time.time()

    # Test: Testing Set       
    acc_num = np.zeros(1).astype('int32')
    loss_test = np.zeros(9999)
    acc_test = np.zeros(9999)
    number = len(Dir_test)
    num_block = 1
    times = round( number / num_block )

    for i in range(0,times):
        
        number = 0
        for j in range(0,num_block):      
            data = h5.File( os.path.join( Dir_test[i*num_block+j] ) , 'r+' )
            tmp_x = np.array( data['im'] )
            tmp_y = np.array( data['la'] )
            data.close()
            length = tmp_x.shape[0]
            batch_x[number:number+length,:,:] = np.concatenate([tmp_x[:,:,:,0], tmp_x[:,:,:,1], tmp_x[:,:,:,2], tmp_x[:,:,:,3]], 2)
            batch_y[number:number+length] = np.argmax( tmp_y , 1 )
            number += length
                             
        history_test = LSTM.evaluate(batch_x[:number,:,:], batch_y[:number], batch_size=1024, verbose=0)    
        loss_test[acc_num] = history_test[0]
        acc_test[acc_num] = history_test[1]
        acc_num += 1
        
    acc_num = acc_num.max()
    loss_test = loss_test[0:acc_num]
    acc_test = acc_test[0:acc_num]
    print( "Iter " + str(step) +", Test Loss= " + "{:.6f}".format(np.mean(loss_test)) + \
          ", Test Accuracy= " + "{:.5f}".format(np.mean(acc_test[0:acc_num])) + \
          ", Time= " + "{:.2f}".format(time.time()-start)) 
    
    if step > 1:
        if np.abs(before-np.mean(loss_test)) < 1e-5:
            flag1 += 1
        else:
            flag1 = 0
        if np.mean(loss_test) > before:
            flag2 += 1
        else: 
            flag2 = 0
        if flag1 == 5 or flag2 ==5:
            break
        
    before = np.mean(loss_test)
    step += 1
    
print( "Optimization Finished!" )

