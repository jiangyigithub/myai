import torch
import torch.nn as tnn
import torchvision.datasets as dsets
import torchvision.transforms as transforms

import os

from torch.utils.data import Dataset
from PIL import Image

import pandas as pd

#change the below parameters accordingly
BATCH_SIZE = 10
LEARNING_RATE = 0.01
EPOCH = 6
N_CLASSES = 43
Pretrained = True
layer_lock = False
train_model = False
test_model = True

use_own_network = True

test_dir= "/fs/scratch/XCSERVER_AI-Initiative/traffic-sign-classification/test_images"
train_dir = "/fs/scratch/XCSERVER_AI-Initiative/traffic-sign-classification/train_images"

#pls using torch.cuda.is_available() to check if .cuda() could be used
#%% to get test data
class my_data(Dataset):
    def __init__(self, image_path, annotation_path=None, transform=None):
        images = os.listdir(image_path)
        self.images = [os.path.join(image_path,image) for image in images]
        self.annotation_path = annotation_path
        self.transform = transform
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, id):
        data=Image.open(self.images[id])
        if self.transform:
            data = self.transform(data)
        return data
        
#%% data preprocessing
transform = transforms.Compose([
    transforms.Resize([224,224]),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean = [ 0.5, 0.5, 0.5 ],
                         std  = [ 0.5, 0.5, 0.5 ]),
    ])

transform_test = transforms.Compose([
    transforms.Resize([224,224]),
    transforms.ToTensor(),
    transforms.Normalize(mean = [ 0.5, 0.5, 0.5 ],
                         std  = [ 0.5, 0.5, 0.5 ]),
    ])
#%% loading data to dataloader
trainData = dsets.ImageFolder(train_dir, transform)
my_data_test = my_data(test_dir, transform=transform_test)

trainLoader = torch.utils.data.DataLoader(dataset=trainData, batch_size=BATCH_SIZE, shuffle=True)
testLoader = torch.utils.data.DataLoader(dataset=my_data_test, batch_size=1, shuffle=False)

#%%  your network
def conv_layer(chann_in, chann_out, k_size, p_size):
    layer = tnn.Sequential(
        tnn.Conv2d(chann_in, chann_out, kernel_size=k_size, padding=p_size),
        tnn.BatchNorm2d(chann_out),
        tnn.ReLU()
    )
    return layer

def vgg_conv_block(in_list, out_list, k_list, p_list, pooling_k, pooling_s):

    layers = [ conv_layer(in_list[i], out_list[i], k_list[i], p_list[i]) for i in range(len(in_list)) ]
    layers += [ tnn.MaxPool2d(kernel_size = pooling_k, stride = pooling_s)]
    return tnn.Sequential(*layers)

def vgg_fc_layer(size_in, size_out):
    layer = tnn.Sequential(
        tnn.Linear(size_in, size_out),
        tnn.BatchNorm1d(size_out),
        tnn.ReLU()
    )
    return layer

class VGG16(tnn.Module):
    def __init__(self, n_classes=1000):
        super(VGG16, self).__init__()

        # Conv blocks (BatchNorm + ReLU activation added in each block)
        self.layer1 = vgg_conv_block([3,64], [64,64], [3,3], [1,1], 2, 2)
        self.layer2 = vgg_conv_block([64,128], [128,128], [3,3], [1,1], 2, 2)
        self.layer3 = vgg_conv_block([128,256,256], [256,256,256], [3,3,3], [1,1,1], 2, 2)
        self.layer4 = vgg_conv_block([256,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)
        self.layer5 = vgg_conv_block([512,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)

        # FC layers
        self.layer6 = vgg_fc_layer(7*7*512, 4096)
        self.layer7 = vgg_fc_layer(4096, 4096)

        # Final layer
        self.layer8 = tnn.Linear(4096, n_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        vgg16_features = self.layer5(out)
        out = vgg16_features.view(out.size(0), -1)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)

        return vgg16_features, out

#%%
if use_own_network:
    vgg16 = VGG16(n_classes=len(trainLoader.dataset.classes))
    vgg16.cuda()

#%%  using opensource models
if not use_own_network:
    
    ## models include Alexnet, Densenet, GoogLeNet, Inception, Mobilenet, VGG, Resnet and so on
    ## choose what you'd like to use
    from torchvision import models  
    
    vgg16=models.vgg16(pretrained=False, num_classes=len(trainLoader.dataset.classes))  ##using original weights pls set pretrained to True and comment pretrained part below
    vgg16.cuda()

#%% Pretrained
if Pretrained:
    state_dict = torch.load("vgg16.pkl")
    #state_dict = torch.load("vgg16.pkl", map_location=torch.device('cpu'))  ## for cpu 
    vgg16.load_state_dict({k:v for k, v in state_dict.items() if k in vgg16.state_dict()})

#%% layer lock
if layer_lock:
    freeze_list =[]
    for p in vgg16.named_parameters():
        freeze_list.append(p[0])
          
    for p in vgg16.named_parameters():
        p[1].requires_grad = False
        if p[0] in freeze_list[-2:]:
            p[1].requires_grad=True

#%%  Loss, Optimizer & Scheduler
# adjust your loss or optimizer as your like,CrossEntropy and Adam is good enough

cost = tnn.CrossEntropyLoss()  
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, vgg16.parameters()), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

#%%
# Train the model
if train_model:
  for epoch in range(EPOCH):

    avg_loss = 0
    cnt = 0
    for images, labels in trainLoader:
        images = images.cuda(0)
        labels = labels.cuda(0)

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        _, outputs = vgg16(images)  #adjust this part if you are using model.vgg16()
        loss = cost(outputs, labels)
        avg_loss += loss.data
        cnt += 1
        print("[E: %d] loss: %f, avg_loss: %f" % (epoch, loss.data, avg_loss/cnt))
        loss.backward()
        optimizer.step()
    scheduler.step(avg_loss)

  #%% Save the Trained Model
  torch.save(vgg16.state_dict(), './vgg16.pkl')
  

#%% Test the model
if test_model:
  vgg16.eval()
  correct = 0
  total = 0
  predicted_score =[]
  file_list = []
  
  for images in testLoader:
        images = images.cuda()
        _, outputs = vgg16(images)  #adjust this part if you are using model.vgg16()

        predicted = int(torch.max(outputs.data, 1).indices)
        predicted_score.append(predicted)
    
        file = testLoader.dataset.images[total]
        file_list.append(int(file[-9:-4]))
        
        print(testLoader.dataset.images[total])
        print("predicted:", predicted)
        
        total +=1
  print(total)

  #save the output
  output = pd.DataFrame({'id':file_list,'pred':predicted_score})
  output = output.sort_values(by=['id'],ascending=True)
  output.to_csv("output.csv",index=False)  


    
  