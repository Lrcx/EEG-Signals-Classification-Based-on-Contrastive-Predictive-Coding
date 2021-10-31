import torch
import numpy as np
train_data=torch.load("train.pt")
# train_data["samples"].squeeze(1)
# train_data["labels"]
val_data=torch.load("val.pt")
test_data=torch.load("test.pt")

data=torch.cat([train_data["samples"],val_data["samples"],test_data["samples"]],dim=0)
label=torch.cat([train_data["labels"],val_data["labels"],test_data["labels"]],dim=0)
print(data.shape,label.shape)

random=np.random.permutation(data.shape[0])
data=data[random]
label=label[random]
total_data={"samples":data,"labels":label}
torch.save(total_data,"total_data.pt")

total_num=[0,0,0,0,0]
for i in label:
    total_num[i]+=1
print(total_num)