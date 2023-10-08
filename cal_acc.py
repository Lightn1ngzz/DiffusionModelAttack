import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import os
import torch.utils.data.distributed
import torchvision.transforms as transforms
import re
from torch.autograd import Variable
import os
from PIL import Image
import linecache



transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

model = models.resnet50(pretrained=False)
fc_inputs = model.fc.in_features
model.fc = nn.Linear(fc_inputs, 14)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model.load_state_dict(torch.load('/root/autodl-tmp/hand_test/best.pt'))
model = model.eval()
model = model.to(device)

path='/root/autodl-tmp/test/output_changelabel_test/clean_images/'
testList=os.listdir(path)
label_path= '/root/autodl-tmp/test_demo/labels.txt'
i = 0
total_number = 0
for file in testList:
    #img=Image.open(path+file)
    img=Image.open(path+file).convert('RGB')
    img=transform_test(img)
    img.unsqueeze_(0)
    img = Variable(img).to(device)
    out=model(img)
   
    # Predict
    pred = torch.argmax(out.data, 1).item()
    #print(file , ':', pred)
    
    # 使用正则表达式提取数字部分
    match = re.search(r'\d+', file)
    
    if match:
        line_number = match.group()
    else:
        print("未找到数字")
    line_number = int(line_number)

    true_label = linecache.getline(label_path, line_number+1)

    true_label=int(true_label)
    #print(true_label)
    total_number+=1
    if pred == true_label:
        i+=1
        #print('right!!!!!')
        
pred_acc = i/total_number
print('acc is: ', pred_acc)
    
    
