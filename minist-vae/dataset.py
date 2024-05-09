from torch.utils.data import Dataset
from torchvision.transforms.v2 import PILToTensor,Compose
import torchvision



import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

class MNIST(Dataset):
    def __init__(self,is_train=True):
        super().__init__()
        self.minist=torchvision.datasets.MNIST('./mnist/',train=is_train,download=True)
        self.img_convert=Compose([
            PILToTensor(),
        ])
        
    def __len__(self):
        return len(self.minist)
    
    def __getitem__(self,index):
        img,label=self.minist[index]
        return self.img_convert(img)/255.0,label
    
if __name__=='__main__':
    import matplotlib.pyplot as plt 
    minist=MNIST()
    img,label=minist[0]
    print(label)
    plt.imshow(img.permute(1,2,0))
    plt.show()