import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import torch.utils.data as data
import torchvision
from torch.autograd import Variable
from functions import *
from mymodel_change_chanel import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score
import pandas as pd
import csv
# set path
data_path = "/home/jfh/Jester-V1/20bn-jester-v1/"
save_model_path ="./train_saver/"
df = pd.read_csv("./jester-v1-labels.csv")
df_train = pd.read_csv("./train_list.csv")
df_validation = pd.read_csv("./val_list.csv")
train_record = './train_record.csv'
val_record = './val_record.csv'



action_names = df['lable'].tolist()
train_file = df_train['train_true'].tolist()
train_y = df_train['true_label'].tolist()

val_file = df_validation['val_true'].tolist()
val_y = df_validation['true_label'].tolist()


res_size = 224        # ResNet image size
dropout_p = 0.3       # dropout probability


# training parameters
k = 27            # number of target category
epochs = 60        # training epochs
batch_size = 16  
learning_rate = 3e-4
log_interval = 50   # interval for displaying training info

# Select which frame to begin & end in videos
begin_frame, end_frame, skip_frame = 1, 8, 1

def save_history(header,value,file_name):
    file_existence = os.path.isfile(file_name)

    # if there is no file make file
    if file_existence == False:
        file = open(file_name, 'w', newline='')
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerow(value)
    # if there is file overwrite
    else:
        file = open(file_name, 'a', newline='')
        writer = csv.writer(file)
        writer.writerow(value)
    # close file when it is done with writing
    file.close()

# Detect devices
use_cuda = torch.cuda.is_available()                   # check if GPU exists
device = torch.device("cuda" if use_cuda else "cpu")   # use CPU or GPU

# Data loading parameters
params = {'batch_size': batch_size, 'shuffle': True, 'num_workers':4, 'pin_memory': True} if use_cuda else {}


# load UCF101 actions names
'''
with open(action_name_path, 'rb') as f:
    action_names = pickle.load(f)
'''

# convert labels -> category
le = LabelEncoder()
le.fit(action_names)

# show how many classes there are
list(le.classes_)

# convert category -> 1-hot
action_category = le.transform(action_names).reshape(-1, 1)
enc = OneHotEncoder()
enc.fit(action_category)

# list all data files
train_list = train_file                  # all video file names
train_label = labels2cat(le, train_y)    # all video labels
test_list = val_file
test_label = labels2cat(le, val_y)
transform = transforms.Compose([transforms.Resize([res_size, res_size]),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

selected_frames = np.arange(begin_frame, end_frame, skip_frame).tolist()

train_set, valid_set = Dataset(data_path, train_list, train_label, selected_frames, transform=transform), \
                       Dataset(data_path, test_list, test_label, selected_frames, transform=transform)

train_loader = data.DataLoader(train_set, **params)
valid_loader = data.DataLoader(valid_set, **params)


# Create model
model = resnet2_3D(num_classes=27, shortcut_type= 'A',sample_size=125, sample_duration=16,last_fc=True).to(device)


# Parallelize model to multiple GPUs
if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)
    params = list(model.parameters())

elif torch.cuda.device_count() == 1:
    print("Using", torch.cuda.device_count(), "GPU!")
    params = list(model.parameters())

#params = list(model.parameters())
optimizer = torch.optim.Adam(params, lr=learning_rate)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.1)

train_header=['epoch','loss','accuracy']
val_header=['epoch','loss','accuracy']
final_accuracy=0
# start training
for epoch in range(epochs):
    #train 
    model.train()
    
    losses = []
    scores = []
    N_count = 0   # counting total trained sample in one epoch
    for batch_idx, (X, y) in enumerate(train_loader):
        # distribute data to device
        X, y = X.to(device), y.to(device).view(-1, )
        N_count += X.size(0)

        optimizer.zero_grad()
        output = model(X)   # output has dim = (batch, number of classes)

        loss = F.cross_entropy(output, y)
        losses.append(loss.item())

        # to compute accuracy
        y_pred = torch.max(output, 1)[1]  # y_pred != output
        step_score = accuracy_score(y.cpu().data.squeeze().numpy(), y_pred.cpu().data.squeeze().numpy())
        scores.append(step_score)         # computed on CPU

        loss.backward()
        optimizer.step()

        # show information
        if (batch_idx + 1) % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accu: {:.2f}%'.format(
                epoch + 1, N_count, len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader), loss.item(), 100 * step_score))
            train_value = [epoch + 1, loss.item(), 100 * step_score]
            # save_history(train_header,train_value,train_record)


    #invalidation
    model.eval() 

    test_loss = 0
    all_y = []
    all_y_pred = []
    with torch.no_grad():
        for X, y in valid_loader:
            # distribute data to device
            X, y = X.to(device), y.to(device).view(-1, )

            output = model(X)

            loss = F.cross_entropy(output, y, reduction='sum')
            test_loss += loss.item()                 # sum up batch loss
            y_pred = output.max(1, keepdim=True)[1]  # (y_pred != output) get the index of the max log-probability

            # collect all y and y_pred in all batches
            all_y.extend(y)
            all_y_pred.extend(y_pred)

    test_loss /= len(valid_loader.dataset)

    # compute accuracy
    all_y = torch.stack(all_y, dim=0)
    all_y_pred = torch.stack(all_y_pred, dim=0)
    test_score = accuracy_score(all_y.cpu().data.squeeze().numpy(), all_y_pred.cpu().data.squeeze().numpy())

    # show and save information
    print('\nTest set ({:d} samples): Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(len(all_y), test_loss, 100* test_score))
    test_value = [epoch+1, test_loss, 100* test_score]
    save_history(val_header,test_value,val_record)

    #save the best model
    if test_score > final_accuracy:
        final_accuracy = test_score
        
        torch.save(model.state_dict(), os.path.join(save_model_path, 'Resnet2D3_longtime.pth'))  # save spatial_encoder
        print("Epoch {} has the best model!".format(epoch + 1))
    #scheduler.step()


