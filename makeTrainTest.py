import ROOT as r 
import numpy as np
import pickle
import math
from keras.utils import np_utils
import tqdm

from multiprocessing import Pool


testCut   = lambda ev: ev.evt%5==0
trainCut  = lambda ev: ev.evt%5!=0



def getVar(ev, l, s):
    return getattr(ev,'LepGood_%s'%s)[l]


featureList = [
    'pt'                   , 
    'eta'                  , 
    'jetNDauChargedMVASel' , 
    'miniRelIsoCharged'    , 
    'miniRelIsoNeutral'    , 
    'jetPtRelv2'           , 
    'jetDeepFlav'          , 
    'jetBTagDeepFlavorlepb', 
    'jetPtRatiov3'         , 
    'dxy'                  , 
    'dz'                   , 
    'sip3d'                , 
]

featureListE = [
    'mvaIdFall17noIso'    
]
featureListM = [
    'segmentCompatibility'  
]




features = {
    'pt'                   : lambda ev, l : ev.LepGood_pt[l],
    'eta'                  : lambda ev, l : ev.LepGood_eta[l] ,
    'jetNDauChargedMVASel' : lambda ev, l : ev.LepGood_jetNDauChargedMVASel[l],
    'miniRelIsoCharged'    : lambda ev, l : ev.LepGood_miniRelIsoCharged[l],
    'miniRelIsoNeutral'    : lambda ev, l : ev.LepGood_miniRelIsoNeutral[l],
    'jetPtRelv2'           : lambda ev, l : ev.LepGood_jetPtRelv2[l],
    'jetDeepFlav'          : lambda ev,l : max(0,ev.LepGood_jetBTagDeepFlavorbb[l]+ev.LepGood_jetBTagDeepFlavorB[l]), 
    'jetBTagDeepFlavorlepb': lambda ev,l : ev.LepGood_jetBTagDeepFlavorlepb[l],
    'jetPtRatiov3'         : lambda ev,l : (ev.LepGood_jetBTagCSV[l]>-5)*min(ev.LepGood_jetPtRatiov2[l],1.5),
    'dxy'                  : lambda ev,l : math.log(abs(ev.LepGood_dxy[l])),
    'dz'                   : lambda ev,l : math.log(abs(ev.LepGood_dz[l])),
    'sip3d'                : lambda ev,l : ev.LepGood_sip3d[l],
    'mvaIdFall17noIso'     : lambda ev,l : ev.LepGood_mvaIdFall17noIso[l],
    'segmentCompatibility' : lambda ev,l : ev.LepGood_segmentCompatibility[l],
    }

classes = {
    'prompt'     : { 'cut': lambda ev, l : ev.LepGood_mcMatchId[l]!=0 and ev.LepGood_mcMatchTau[l]==0, 'lst_E_train' : [], 'lst_E_test' : [] , 'lst_M_train' : [], 'lst_M_test' : [], 'lst_E_y_train' : [], 'lst_E_y_test' : [] , 'lst_M_y_train' : [], 'lst_M_y_test' : []},
    'prompt_tau' : { 'cut': lambda ev, l : ev.LepGood_mcMatchId[l]!=0 and ev.LepGood_mcMatchTau[l]==1, 'lst_E_train' : [], 'lst_E_test' : [] , 'lst_M_train' : [], 'lst_M_test' : [], 'lst_E_y_train' : [], 'lst_E_y_test' : [] , 'lst_M_y_train' : [], 'lst_M_y_test' : []},
    'hfake'      : { 'cut': lambda ev, l : ev.LepGood_mcMatchId[l]==0 and (abs(ev.LepGood_mcMatchAny[l])==4 or abs(ev.LepGood_mcMatchAny[l])==5) , 'lst_E_train' : [], 'lst_E_test' : [] , 'lst_M_train' : [], 'lst_M_test' : [], 'lst_E_y_train' : [], 'lst_E_y_test' : [] , 'lst_M_y_train' : [], 'lst_M_y_test' : []},
    'lfake'      : { 'cut': lambda ev, l : ev.LepGood_mcMatchId[l]==0 and (abs(ev.LepGood_mcMatchAny[l])<4 or abs(ev.LepGood_mcMatchAny[l])>5), 'lst_E_train' : [], 'lst_E_test' : [] , 'lst_M_train' : [], 'lst_M_test' : [], 'lst_E_y_train' : [], 'lst_E_y_test' : [] , 'lst_M_y_train' : [], 'lst_M_y_test' : []},
}

sampleDir='/pool/ciencias/userstorage/sscruz/HeppyTrees/2018/TREES_TTH_190418_Fall17_fromCarlos_withDF/'

sigStuff = []
bkgStuff = []

bkgStuff.extend( [ "TT_Semilep_powheg_part%d/treeProducerSusyMultilepton/tree.root"%i for i in range(1,16)])
bkgStuff.extend( [ "TT_Dil_powheg_part%d/treeProducerSusyMultilepton/tree.root"%i for i in range(1,25)])


sigStuff.extend( ["TTHnobb_powheg_part%d/treeProducerSusyMultilepton/tree.root"% i for i in range(1,3)])
sigStuff.extend( ["TTHincl_powheg_part%d/treeProducerSusyMultilepton/tree.root"% i for i in range(1,4)])

# # calmarse un poquito, aqui
# bkgStuff = [bkgStuff[0]]
# sigStuff = [sigStuff[0]]

tasks = []


def toNumpy(task):
    print 'starting', task
    fil, typs, ch = task
    print 'List of features for', ch, featureList + eval('featureList%s'%ch)
    tfile = r.TFile(fil); ttree = tfile.tree
    results = {}
    for ty in typs: 
        results[ty + '_test']  = []
        results[ty + '_train'] = []

    for ev in ttree:
        tstr = 'test' if testCut(ev) else 'train'
        for l in range(ev.nLepGood):
            if   ch == 'E':
                if abs(ev.LepGood_pdgId[l]) != 11: continue
            elif ch == 'M': 
                if abs(ev.LepGood_pdgId[l]) != 13: continue
            else: raise RuntimeError("Wrong chan %s"%ch)
            for ty in typs:
                if classes[ty]['cut'](ev,l):
                    results[ty+'_'+tstr].append([ features[s](ev, l) for s in (featureList + eval('featureList%s'%ch)) ])
    tfile.Close()
    print 'finishing', task
    return results, ch


print 'Setting up the tasks'
for samp in sigStuff:
    tasks.append( (sampleDir+'/'+samp, ['prompt','prompt_tau'], 'E') )
    tasks.append( (sampleDir+'/'+samp, ['prompt','prompt_tau'], 'M') )



for samp in bkgStuff:
    tasks.append( (sampleDir+'/'+samp, ['hfake','lfake'], 'E') )
    tasks.append( (sampleDir+'/'+samp, ['hfake','lfake'], 'M') )
    
                
print 'Going to run the big thing'
p =  Pool(50)
results = list(tqdm.tqdm(p.imap( toNumpy, tasks), total=len(tasks)))


print 'Now putting everything together'

types = ['prompt', 'prompt_tau', 'hfake', 'lfake']
for result, ch in results: 
    for ty in types:
        if ty+'_train' in result:  # only some "results" have prompt / fake
            classes[ty]['lst_%s_train'%ch].extend( result[ty+'_train'])
            classes[ty]['lst_%s_test'%ch ].extend( result[ty+'_test'])

            
print 'Setting the indices'
    
toDump = {} 

for ch in 'E,M'.split(','):
    for i, ty in enumerate(types):
        classes[ty]['lst_%s_train'%ch  ] = np.asarray(classes[ty]['lst_%s_train'%ch])
        classes[ty]['lst_%s_y_train'%ch] = i*np.ones((classes[ty]['lst_%s_train'%ch].shape[0],1))
        classes[ty]['lst_%s_test'%ch   ] = np.asarray(classes[ty]['lst_%s_test'%ch])
        classes[ty]['lst_%s_y_test'%ch ] = i*np.ones((classes[ty]['lst_%s_test'%ch].shape[0],1))
    train_x = np.concatenate( tuple( [classes[ty]['lst_%s_train'%ch] for ty in types] ), axis=0)
    train_y = np_utils.to_categorical( np.concatenate( tuple( [classes[ty]['lst_%s_y_train'%ch] for ty in types] ), axis=0), len(classes))
    test_x = np.concatenate( tuple( [classes[ty]['lst_%s_test'%ch] for ty in types] ), axis=0)
    test_y = np_utils.to_categorical( np.concatenate( tuple( [classes[ty]['lst_%s_y_test'%ch] for ty in types] ), axis=0), len(classes))


    toDump['train_%s_x'%ch] = train_x
    toDump['train_%s_y'%ch] = train_y
    toDump['test_%s_x' %ch] = test_x 
    toDump['test_%s_y' %ch] = test_y 

pickle.dump( toDump, open('vars.p','w'))
