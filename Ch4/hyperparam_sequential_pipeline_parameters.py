import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import kfp
from kfp.components import func_to_container_op
import kfp_tekton
import config, utils

#figure out parallelfor, for dynamic loops

'''
Cartoon for random search

Start at random point
evaluate f

loop:
    pick point
    evaluate f
    if f better: update val

'''
BASE_IMAGE = config.BASE_IMAGE
S3_END_POINT = config.S3_END_POINT
S3_ACCESS_ID = config.S3_ACCESS_ID
S3_ACCESS_KEY = config.S3_ACCESS_KEY
bucket_name = config.BUCKET_NAME

get_client = utils.get_client
create_bucket = utils.create_bucket
read_from_store = utils.read_from_store
write_to_store = utils.write_to_store

class Net(nn.Module):
    def __init__(self, n_inputs=2, n_outputs=1, n_hidden_nodes=10, n_hidden_layers=1, activation=nn.ReLU(), output_activation=None):
        super(Net, self).__init__()

        self.layer_list = nn.ModuleList()

        for i in range(n_hidden_layers):
            if i==0:
                self.layer_list.append(nn.Linear(n_inputs, n_hidden_nodes))
            else:
                self.layer_list.append(nn.Linear(n_hidden_nodes, n_hidden_nodes))
        
        if n_hidden_layers==0:
            self.output_layer = nn.Linear(n_inputs, n_outputs)
        else:
            self.output_layer = nn.Linear(n_hidden_nodes, n_outputs)

        self.activation = activation
        self.output_activation = output_activation

    def forward(self, x):
        out = x

        for layer in self.layer_list:
            out = self.activation(layer(out))

        out = self.output_layer(out)
        if self.output_activation is not None:
            out = self.output_activation(out)

        return out

def fit_gp_model():
    #read data from store

    #fit gp model

    #compute acquisition function

    #maximize acquisition function

    #write point to store
    pass

def generate_random_search_point(iter_val: int) -> int:
    #read domain from store

    #generate random neighbor

    #write to store
    client = get_client()
    print(iter_val)

    if iter_val==0:
        conf = {
            'lr': 1e-2,
            'num_hidden_layers': 1,
            'num_nodes': 1,
            'activation': 'relu'
        }
        write_to_store(bucket_name, conf, f'conf_{iter_val}', client)
        print(conf)
        return iter_val

    conf = read_from_store(bucket_name, f'conf_{iter_val-1}', client)
    
    #transition for num_nodes: -1/0/+1 with num_nodes >= 1
    conf['num_nodes'] = max(conf['num_nodes'] + (np.random.randint(3)-1), 1)

    #transition for num_hidden_layers: -1/0/+1 with num_hidden_layers >= 0
    conf['num_hidden_layers'] = max(conf['num_hidden_layers'] + (np.random.randint(3)-1), 1)

    #learning rate: multiply by 0.1/1/10
    conf['lr'] = conf['lr'] * 10**(np.random.randint(3)-1)

    #activation
    conf['activation'] = 'relu' if np.random.uniform() < 0.5 else 'sigmoid'

    write_to_store(bucket_name, conf, f'conf_{iter_val}', client)
    print(conf)

    return iter_val

def print_gen_val(iter_val: int) -> int:
    client = get_client()

    for i in range(iter_val):
        conf = read_from_store(bucket_name, f'score_{i}', client)
        print(i)
        print(conf)
        print('-----------')

    return iter_val

def download_data(numExamples: int, numSeed: int) -> int:
    '''Download and store data in persistent storage
    '''
    return 0

    client = get_client()

    def generate_binary_data(N_examples=1000, seed=None):
    #Generate N_examples points with two features each
    #
    #Args:
    #    seed: seed that should be fixed if want to generate same points again    
    #Returns:
    #    features: A 2-dimensional numpy array with one row per example and one column per feature
    #    target: A 1-dimensional numpy array with one row per example denoting the class - 0 or 1

        if seed is not None:
            np.random.seed(seed)

        features = []
        target = []

        for i in range(N_examples):
            #class = 0
            r = np.random.uniform() #class 0 has radius between 0 and 1
            theta = np.random.uniform(0, 2*np.pi) #class 0 has any angle between 0 and 360 degrees

            features.append([r*np.cos(theta), r*np.sin(theta)])
            target.append(0)

            #class = 1
            r = 3 + np.random.uniform() #class 1 has radius between 3+0=3 and 3+1=4
            theta = np.random.uniform(0, 2*np.pi) #class 1 has any angle between 0 and 360 degrees

            features.append([r*np.cos(theta), r*np.sin(theta)])
            target.append(1)

        features = np.array(features)
        target = np.array(target)

        return features, target

    features_train, target_train = generate_binary_data(N_examples=numExamples, seed=numSeed)
    features_test, target_test = generate_binary_data(N_examples=numExamples+500, seed=numSeed + 10)

    create_bucket(bucket_name, client)
    write_to_store(bucket_name, features_train, 'features_train', client)
    write_to_store(bucket_name, target_train, 'target_train', client)
    write_to_store(bucket_name, features_test, 'features_test', client)
    write_to_store(bucket_name, target_test, 'target_test', client)

    return 0

def train_model(hyperparam_idx: int, retcode_genpoint: int) -> int:
    '''Look up hyperparams from store
    and train model
    '''

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Device = {device}')

    client = get_client()

    #load data from store
    features_train = torch.from_numpy(read_from_store(bucket_name, 'features_train', client)).float()
    target_train = torch.from_numpy(read_from_store(bucket_name, 'target_train', client)).float()
    features_test = torch.from_numpy(read_from_store(bucket_name, 'features_test', client)).float()
    target_test = torch.from_numpy(read_from_store(bucket_name, 'target_test', client)).float()

    #load conf from store
    conf = read_from_store(bucket_name, f'conf_{hyperparam_idx}', client)
    lr = float(conf.get('lr', 1e-2))
    N_epochs = int(conf.get('N_epochs', 10000))
    num_hidden_layers = int(conf.get('num_hidden_layers', 1))
    num_nodes = int(conf.get('num_nodes', 2))
    activation = conf.get('activation', 'relu')

    #should be dependent on vars read from config
    if activation=='relu':
        activation = nn.ReLU()
    elif activation=='sigmoid':
        activation = nn.Sigmoid()

    #initialize model
    model = Net(n_inputs=2, n_outputs=1, n_hidden_nodes=num_nodes, n_hidden_layers=num_hidden_layers, activation=activation, output_activation=nn.Sigmoid())
    
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) #Adam optimizer
    model.train()    

    print(hyperparam_idx)
    print(conf)

    if device!='cpu':
        model = model.to(device)
        features_train = features_train.to(device)
        target_train = target_train.to(device)

    for epoch in range(N_epochs): #N_epochs = number of iterations over the full dataset
        features_shuffled = features_train
        target_shuffled = target_train

        out = model(features_shuffled) #predictions from model
        loss = criterion(out.squeeze(), target_shuffled.squeeze()) #loss between predictions and labels

        if epoch % 1000 == 0:
            print(f'epoch = {epoch} loss = {loss}')

        optimizer.zero_grad()
        loss.backward() #compute gradients
        optimizer.step() #update model

    out = model(features_shuffled) #predictions from model
    train_loss = criterion(out.squeeze(), target_shuffled.squeeze()) #loss between predictions and labels
    print(f'Train Loss : {train_loss}')

    def evaluate_model(model, features_test, target_test):
        '''Evaluate model on test set
        and store result
        '''
        model.eval()

        if device!='cpu':
            features_test = features_test.to(device)
            target_test = target_test.to(device)

        out = model(features_test)
        loss = criterion(out.squeeze(), target_test.squeeze())
        

        return loss

    test_loss = evaluate_model(model, features_test, target_test)
    print(f'Test  Loss : {test_loss}')

    write_to_store(bucket_name, {'test_loss': test_loss.item(), 'conf': conf}, f'score_{hyperparam_idx}', client)

    return hyperparam_idx

download_data_op = func_to_container_op(download_data, base_image=BASE_IMAGE, packages_to_install=["boto3"], modules_to_capture=["utils"], use_code_pickling=True)
gen_random_op = func_to_container_op(generate_random_search_point, base_image=BASE_IMAGE, packages_to_install=["boto3"], modules_to_capture=["utils"], use_code_pickling=True)
print_gen_val_op = func_to_container_op(print_gen_val, base_image=BASE_IMAGE, packages_to_install=["boto3"], modules_to_capture=["utils"], use_code_pickling=True)
train_model_op = func_to_container_op(train_model, base_image=BASE_IMAGE, packages_to_install=["boto3"], modules_to_capture=["utils"], use_code_pickling=True)

@kfp.dsl.pipeline(
    name='Full pipeline'
)

def run_pipeline(rangeValue: int = 1000, seedValue: int = 100):
    retcode_download = download_data_op(rangeValue, seedValue)
    
    retcode_placeholder = retcode_download
    
    for i in list(range(10)): #try to be as close as possible to canonical python for DS people
        retcode_placeholder = gen_random_op(i).after(retcode_placeholder)

        #retcode_placeholder = print_gen_val_op(retcode_placeholder.output)
        retcode_placeholder = train_model_op(i, retcode_placeholder.output)

    print_gen_val_op(10).after(retcode_placeholder)
#------------------------------------------------------

#kfp.compiler.Compiler().compile(run_pipeline, 'test_nested.yaml')
from kfp_tekton.compiler import TektonCompiler
TektonCompiler().compile(run_pipeline, 'test_sequential_param.yaml')

'''
Q: can ops be wrapped in other ops

op - deferred till pipeline executed
non-op - run now

https://kubeflow-pipelines.readthedocs.io/en/latest/source/kfp.dsl.html

BaseOp
Condition
ContainerOp
ExitHandler
InputArgumentPath
ParallelFor
PipelineConf
PipelineExecutionMode
PipelineParam
PipelineVolume
ResourceOp
Sidecar
SubGraph
UserContainer
VolumeOp
VolumeSnapshotOp
'''
