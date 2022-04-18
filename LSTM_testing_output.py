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
    
test_D = 'F:/Work2/EEG/SHHS/test/'
test_O = 'F:/Work2/EEG/results/SHHS/test/'

# Cost
LEARNING_RATE = 0.001
LSTM = tf.keras.models.load_model('E:/Work/model-N1-SHHS/'+"%03d"%(120))              
LSTM.compile(optimizer=tf.keras.optimizers.Adam(LEARNING_RATE), \
             loss='sparse_categorical_crossentropy', \
             metrics=['accuracy'])  
LSTM.summary()

# Keep training until reach max iteration 
num_block = 1
batch_x = np.zeros([4096,96,128],dtype='float32')
batch_y = np.zeros([4096],dtype='float32')

# Keep training until reach max iterations 
Dir_test = os.listdir( test_D )
number = len(Dir_test)
num_block = 1
times = round( number / num_block )

start = time.time()
for i in range(0,times):
    
    number = 0
    for j in range(0,num_block):      

        data = h5.File( os.path.join( test_D , Dir_test[i*num_block+j] ) , 'r+' , rdcc_nbytes=1073741824 )
        tmp_x = np.array( data['im'] )
        tmp_y = np.array( data['la'] )
        data.close()
        length = tmp_x.shape[0]
        batch_x[number:number+length,:,:] = np.concatenate([tmp_x[:,:,:,0], tmp_x[:,:,:,1], tmp_x[:,:,:,2], tmp_x[:,:,:,3]], 2)
        batch_y[number:number+length] = np.argmax( tmp_y , 1 )
        number += length
                                   
    pathout = test_O
    if os.path.exists(pathout) == 0:
        os.makedirs(pathout)
    tmp = LSTM.predict(batch_x[:number,:,:], batch_size = 1024, verbose=0)
    np.savetxt(pathout+'%02d'%(i)+'.txt',tmp)
print(pathout+'   %.2f'%(time.time()-start))