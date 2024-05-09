from torch.utils.data import Dataset
from torchvision.transforms.v2 import PILToTensor,Compose
import torchvision
from torchvision import transforms 
from config import *

# PIL图像转tensor
pil_to_tensor=transforms.Compose([
    transforms.Resize((IMG_SIZE,IMG_SIZE)),    # PIL图像尺寸统一  
    transforms.ToTensor()                       # PIL图像转tensor, (H,W,C) ->（C,H,W）,像素值[0,1]
])

# tensor转PIL图像
tensor_to_pil=transforms.Compose([
    transforms.Lambda(lambda t: t*255),  # 像素还原
    transforms.Lambda(lambda t: t.type(torch.uint8)),    # 像素值取整
    transforms.ToPILImage(),    # tensor转回PIL图像, (C,H,W) -> (H,W,C) 
])

class MNIST(Dataset):
    def __init__(self,is_train=True):
        super().__init__()
        self.ds=torchvision.datasets.MNIST('./mnist/',train=is_train,download=True)
        self.img_convert=Compose([
            PILToTensor(),
        ])
        
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self,index):
        img,label=self.ds[index]
        return self.img_convert(img)/255.0,label
    
if __name__=='__main__':
    import matplotlib.pyplot as plt 
    
    ds=MNIST()
    img,label=ds[0]
    print(label)
    plt.imshow(img.permute(1,2,0))
    plt.show()