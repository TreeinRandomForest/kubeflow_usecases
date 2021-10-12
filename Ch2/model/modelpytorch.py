#import boto3
import json
import os
import torch




class modelpytorch(object):
 
    def __init__(self):
        print("Initializing")
        #///////// Loading mode from filesystem
        self.clf = torch.load('modelpytorch.pt')
        print("Model uploaded to class")
    
    def predict(self,x):
        print(x)
        print(type(x))
        #old way
        #featurearray=[float(i) for i in x.split(',')]
        #print(featurearray)
        #rowdf = pd.DataFrame([featurearray], columns = ['input1','input2'])
        #print(rowdf)
        #self.proba_1 = self.clf.predict_proba(rowdf)[:,1]
        #predictions = self.clf.predict(rowdf)
        #return predictions
        features = torch.from_numpy(x).float()
        pred = self.clf(features)
        
        return pred.detach().numpy()
