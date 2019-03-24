import numpy as np
import pickle
import matplotlib
matplotlib.use('ps')
import matplotlib.pyplot as plt
import pandas #Scatter plotting
import seaborn as sns #Densities plotting

from pandas.plotting import scatter_matrix
from keras.models import load_model
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

data = pickle.load( open('/nfs/fanae/user/sscruz/TTH/forDeepFlav/CMSSW_9_4_4/src/CMGTools/TTHAnalysis/macros/leptons/multiclass/vars.p','rb'))
data2 = pickle.load( open('/nfs/fanae/user/sscruz/TTH/forDeepFlav/CMSSW_9_4_4/src/CMGTools/TTHAnalysis/macros/leptons/multiclass/vars_onlyLepMVA_toEval.p','rb'))

# fig, ax = plt.subplots()
# plt.hist(data['train_x'])
# fig.savefig('hist.png')



names =  np.array(['pt','eta','jetNDauCHargedMVASel','miniRelIsoCharged','miniRelIsoNeutral','jetPtRelv2','jetDeepFlav','jetBTagDeepFlavorlepb','jetPtRatiov3','dxy','dz','sip3d','12'])
names2 = np.array(['prompt','prompt tau','heavy fake','light fake'])
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
    auc_old = auc(fpr_old, tpr_old)

    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
    plt.plot(fpr_old, tpr_old, label='BDT (area = {:.3f})'.format(auc_old))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')

    plt.show()
    plt.savefig('roc_%s.png'%ch)

    yy = np.reshape(y,(y.size,1))
    stogether = np.append(x,yy,axis=1)
    stogether2 = np.append(prediction,yy,axis=1)
#    plt.clf()
#    sns.pairplot(pandas.DataFrame(stogether,columns=np.append(names,'type')),hue='type') #It takes time
#    plt.show()
#    plt.savefig('Pairplot_%s'%(ch))

    ax_y = ['linear']*13 #Solution to logarithmic scale (default is linear)
    for var in [2,3,4,5,7]: #Plots with logarithmic axis y
    	ax_y[var] = 'log'

    for var in range(13):
#        print('%s %d'%(ch,var)) #To know the stage of the script
        together = np.dstack( (x[:,var], y ) )[0]
        
        class1 =  (together[together[:,1] == 0]) [:,0]
        class2 =  (together[together[:,1] == 1]) [:,0]
        class3 =  (together[together[:,1] == 2]) [:,0]
        class4 =  (together[together[:,1] == 3]) [:,0]

	classglobal = np.concatenate((class1, class2, class3, class4)) #Solution to bins slightly shifted
	xmax = max(classglobal)
	xmin = min(classglobal)

        bins = 20
        plt.clf()
	matplotlib.style.use('classic')
        plt.hist(class1, np.linspace(xmin, xmax, bins+1), alpha=0.8,normed=True, label='Prompt', histtype = 'step')
        plt.hist(class2, np.linspace(xmin, xmax, bins+1), alpha=0.8,normed=True, label='Prompt tau', histtype = 'step')
        plt.hist(class3, np.linspace(xmin, xmax, bins+1), alpha=0.8,normed=True, label='Heavy fake', histtype = 'step')
        plt.hist(class4, np.linspace(xmin, xmax, bins+1), alpha=0.8,normed=True, label='Light fake', histtype = 'step')
        plt.legend(loc='upper right')

        plt.legend(loc='upper right')
        plt.show()
	plt.yscale(ax_y[var])
    
        plt.savefig('input_%s_%s.png'%(ch,names[var]))

        plt.clf()
        sns.kdeplot(class1,shade=True,label='Prompt')
        sns.kdeplot(class2,shade=True,label='Prompt tau')
        sns.kdeplot(class3,shade=True,label='Heavy fake')
        sns.kdeplot(class4,shade=True,label='Light fake')

	plt.show()
	plt.savefig('inputdensity_%s_%s.png'%(ch,names[var]))

    for node in range(4):
        together = np.dstack( (prediction[:,node], y ) )[0]
        
        class1 =  (together[together[:,1] == 0]) [:,0]
        class2 =  (together[together[:,1] == 1]) [:,0]
        class3 =  (together[together[:,1] == 2]) [:,0]
        class4 =  (together[together[:,1] == 3]) [:,0]

	classglobal = np.concatenate((class1, class2, class3, class4))
	xmax = max(classglobal)
	xmin = min(classglobal)
        
        bins = 20
        plt.clf()
        plt.hist(class1, np.linspace(xmin, xmax, bins+1), alpha=1,normed=True, label='Prompt', histtype = 'step')
        plt.hist(class2, np.linspace(xmin, xmax, bins+1), alpha=1,normed=True, label='Prompt tau', histtype = 'step')
        plt.hist(class3, np.linspace(xmin, xmax, bins+1), alpha=1,normed=True, label='Heavy fake', histtype = 'step')
        plt.hist(class4, np.linspace(xmin, xmax, bins+1), alpha=1,normed=True, label='Light fake', histtype = 'step')
        plt.legend(loc='upper right')
        plt.show()

        plt.savefig('output_%s_%s.png'%(ch,names2[node]))
        df=pandas.DataFrame((stogether2[stogether2[:,4]==node])[:,:-1],columns=names2)
        scatter_matrix(df) 
        plt.show()
        plt.savefig('ScatterOutput_%s_%s.png'%(ch,names2[node]))

	
    together = np.dstack( (classifier,y))[0]
    class1 =  (together[together[:,1] == 0]) [:,0]
    class2 =  (together[together[:,1] == 1]) [:,0]
    class3 =  (together[together[:,1] == 2]) [:,0]
    class4 =  (together[together[:,1] == 3]) [:,0]
    classglobal = np.concatenate((class1, class2, class3, class4))
    xmax = max(classglobal)
    xmin = min(classglobal)
    bins = 20
    plt.clf()
    plt.hist(class1, np.linspace(xmin, xmax, bins+1), alpha=1,normed=True, label='Prompt', histtype = 'step', fill = None)
    plt.hist(class2, np.linspace(xmin, xmax, bins+1), alpha=1,normed=True, label='Prompt tau', histtype = 'step', fill = None)
    plt.hist(class3, np.linspace(xmin, xmax, bins+1), alpha=1,normed=True, label='Heavy fake', histtype = 'step', fill = None)
    plt.hist(class4, np.linspace(xmin, xmax, bins+1), alpha=1,normed=True, label='Light fake', histtype = 'step', fill = None)
    plt.legend(loc='upper right')
    plt.show()
    
    plt.savefig('output_%s_combined.png'%(ch))
        
# count = 0
# for i in together:
#     print i[0], i[1]
#     count = count +1 

