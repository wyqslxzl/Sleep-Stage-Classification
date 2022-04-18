# Sleep-Stage-Classification
 The LSTM network training and testing codes
Network training and Testing Code. The input data is a 4x32x96x4096 matric (named 'im'). After read by hd5 function, this matrix will be transposed into 4096x96x32x4, and the code will automatically change into 4096x96x128 matrix. The input label is a 5x4096 (named 'la').
