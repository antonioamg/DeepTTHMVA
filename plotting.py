import numpy as np
import pickle
import matplotlib
matplotlib.use('ps')
import matplotlib.pyplot as plt

from keras.models import load_model
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

data = pickle.load( open('/nfs/fanae/user/sscruz/TTH/forDeepFlav/CMSSW_9_4_4/src/CMGTools/TTHAnalysis/macros/leptons/multiclass/vars.p','rb'))
data2 = pickle.load( open('/nfs/fanae/user/sscruz/TTH/forDeepFlav/CMSSW_9_4_4/src/CMGTools/TTHAnalysis/macros/leptons/multiclass/vars_onlyLepMVA_toEval.p','rb'))

# fig, ax = plt.subplots()
# plt.hist(data['train_x'])
# fig.savefig('hist.png')


for ch in 'E,M'.split(','):
    plt.clf()
    y2 = np.argmax( data2['test_%s_y'%ch], axis=1)
    x2 = np.sum( (data2['test_%s_x'%ch])[:,[0]], axis=1)
    fpr_old, tpr_old, thresholds_old = roc_curve(y2<=1, x2)

    model = load_model('trained_model_B_%s.h5'%ch)

    prediction = model.predict(data['test_%s_x'%ch])
    x = data['test_%s_x'%ch]
    y = np.argmax(data['test_%s_y'%ch], axis=1)

    classifier = np.sum( prediction[:,[0,1]], axis=1)


    fpr_keras, tpr_keras, thresholds_keras = roc_curve(y<=1, classifier)
    auc_keras = auc(fpr_keras, tpr_keras)

    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
    plt.plot(fpr_old, tpr_old, label='Keras (area = ??)')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')

    plt.show()
    plt.savefig('roc_%s.png'%ch)


    for var in range(13):
        together = np.dstack( (x[:,var], y ) )[0]
        
        class1 =  (together[together[:,1] == 0]) [:,0]
        class2 =  (together[together[:,1] == 1]) [:,0]
        class3 =  (together[together[:,1] == 2]) [:,0]
        class4 =  (together[together[:,1] == 3]) [:,0]

        bins = 20
        plt.clf()
        plt.hist(class1, bins, alpha=0.5,normed=True, label='Prompt')
        plt.hist(class2, bins, alpha=0.5,normed=True, label='Prompt tau')
        plt.hist(class3, bins, alpha=0.5,normed=True, label='Heavy fake')
        plt.hist(class4, bins, alpha=0.5,normed=True, label='Light fake')
        plt.legend(loc='upper right')
        plt.show()
    
        plt.savefig('input_%s_%d.png'%(ch,var))


    for node in range(4):
        together = np.dstack( (prediction[:,node], y ) )[0]
        
        class1 =  (together[together[:,1] == 0]) [:,0]
        class2 =  (together[together[:,1] == 1]) [:,0]
        class3 =  (together[together[:,1] == 2]) [:,0]
        class4 =  (together[together[:,1] == 3]) [:,0]
        
        bins = 20
        plt.clf()
        plt.hist(class1, bins, alpha=0.5,normed=True, label='Prompt')
        plt.hist(class2, bins, alpha=0.5,normed=True, label='Prompt tau')
        plt.hist(class3, bins, alpha=0.5,normed=True, label='Heavy fake')
        plt.hist(class4, bins, alpha=0.5,normed=True, label='Light fake')
        plt.legend(loc='upper right')
        plt.show()

        plt.savefig('output_%s_%d.png'%(ch,node))

    together = np.dstack( (classifier,y))[0]
    class1 =  (together[together[:,1] == 0]) [:,0]
    class2 =  (together[together[:,1] == 1]) [:,0]
    class3 =  (together[together[:,1] == 2]) [:,0]
    class4 =  (together[together[:,1] == 3]) [:,0]
    bins = 20
    plt.clf()
    plt.hist(class1, bins, alpha=0.5,normed=True, label='Prompt')
    plt.hist(class2, bins, alpha=0.5,normed=True, label='Prompt tau')
    plt.hist(class3, bins, alpha=0.5,normed=True, label='Heavy fake')
    plt.hist(class4, bins, alpha=0.5,normed=True, label='Light fake')
    plt.legend(loc='upper right')
    plt.show()
    
    plt.savefig('output_%s_combined.png'%(ch))
        

# count = 0
# for i in together:
#     print i[0], i[1]
#     count = count +1 

