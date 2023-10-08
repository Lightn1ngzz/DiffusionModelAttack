import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
from Data import hand_pose
from Model import ResNet50, Bottleneck
import os
from torchvision import models
import matplotlib.pyplot as plt
import numpy as np

def main():
    # 1. load dataset
    root = "/root/autodl-tmp/train_adv"
    batch_size = 64
    train_data = hand_pose(root, train=True)
    val_data = hand_pose(root, train=False)
    train_dataloader = DataLoader(train_data,batch_size=batch_size,shuffle=True)
    val_dataloader = DataLoader(val_data,batch_size=batch_size,shuffle=True)
    
    # # 2. load model
    # num_class = 14
    # model = ResNet50(Bottleneck,[3,4,6,3], num_class)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = model.to(device)
    
    # 2. load model
    #num_class = 14
    ## model = ResNet50(Bottleneck,[3,4,6,3], num_class)
    #model = models.resnet50(pretrained=True)
    #fc_inputs = model.fc.in_features
    #model.fc = nn.Linear(fc_inputs, num_class)
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #model = model.to(device)
    # 2. load model
    model = models.resnet50(pretrained=False)
    fc_inputs = model.fc.in_features
    model.fc = nn.Linear(fc_inputs, 14)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    model.load_state_dict(torch.load('/root/autodl-tmp/hand_test/before_best.pt'))
    model = model.to(device)
    # 3. prepare super parameters
    criterion = nn.CrossEntropyLoss()
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    epoch = 40
 
    # 4. train
    val_acc_list = []
    acclist = []
    for epoch in range(0, epoch):
        print('\nEpoch: %d' % (epoch + 1))
        model.train()
        sum_loss = 0.0
        correct = 0.0
        total = 0.0
        for batch_idx, (images, labels) in enumerate(train_dataloader):
            length = len(train_dataloader)
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images) # torch.size([batch_size, num_class])
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
            sum_loss += loss.item()
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += predicted.eq(labels.data).cpu().sum()
            print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% ' 
                % (epoch + 1, (batch_idx + 1 + epoch * length), sum_loss / (batch_idx + 1), 100. * correct / total))
            
        #get the ac with testdataset in each epoch
        print('Waiting Val...')
        with torch.no_grad():
            correct = 0.0
            total = 0.0
            for batch_idx, (images, labels) in enumerate(val_dataloader):
                model.eval()
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum()
            print('Val\'s ac is: %.3f%%' % (100 * correct / total))
            
            acc_val = 100 * correct / total
            val_acc_list.append(acc_val)
            acclist.append(acc_val.item())
 
 
        torch.save(model.state_dict(), "after_last.pt")
        if acc_val == max(val_acc_list):
            torch.save(model.state_dict(), "after_best.pt")
            print("save epoch {} model".format(epoch))
            
    x= list(range(0,len(acclist)))
    plt.plot(x,acclist,'.-',label = 'acc')        # epoch_losses 传入模型训练中的 loss[]列表,在训练过程中，先创建loss列表，将每一个epoch的loss 加进这个列表
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.savefig("acc_after.png") #保存图片 路径：/imgPath/
 
if __name__ == "__main__":
    main()
    
    

