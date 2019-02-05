import ROOT as r 
import numpy as np
import pickle
import math
from keras.models import Sequential
from keras.layers import Dense, Dropout

import tensorflow as tf
from keras.backend import tensorflow_backend as K

from keras import optimizers

data = pickle.load( open('/nfs/fanae/user/sscruz/TTH/forDeepFlav/CMSSW_9_4_4/src/CMGTools/TTHAnalysis/macros/leptons/multiclass/vars.p','rb'))
print data
print kk 

def getCompiledModelA():
    # optimal so far ( 0.980, 0.966)
    model = Sequential()
    model.add(Dense(30,input_dim=13, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(4, activation='softmax'))
    adam = optimizers.adam(lr=1e-4) 
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy','categorical_crossentropy'])
    return model

def getCompiledModelB():
    model = Sequential()
    model.add(Dense(30,input_dim=13, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(10, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(4, activation='softmax'))
    adam = optimizers.adam(lr=1e-4) 
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy','categorical_crossentropy'])
    return model


for ch in 'E,M'.split(','):

    sums = np.sum(data['train_%s_y'%ch],axis=0)
    print sums 
    print kk 

    sig = sums[0] + sums[1]
    bkg = sums[2] + sums[3]

    class_weight = { 0 : float((sig+bkg)/sig),
                     1 : float((sig+bkg)/sig),
                     2 : float((sig+bkg)/bkg),
                     3 : float((sig+bkg)/bkg)}
    print 'weights will be', class_weight

    with tf.Session(config=tf.ConfigProto(
            intra_op_parallelism_threads=50,
            inter_op_parallelism_threads=50)) as sess:
        K.set_session(sess)
        
        #model = getCompiledModelA()
        model = getCompiledModelB()

        history = model.fit( data['train_%s_x'%ch], data['train_%s_y'%ch], epochs=10, batch_size=128, validation_data=(data['test_%s_x'%ch], data['test_%s_y'%ch]), class_weight=class_weight)

        model.save('trained_model_B_%s.h5'%ch)
        pickle.dump( history.history, open('history_B_%s.p'%ch,'w'))
