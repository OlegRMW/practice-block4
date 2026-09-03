import torch.nn as nn 

# выход [64, H, W]
class BasicBlock1(nn.Module): 
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64), 
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64)
        )
        self.shortcut = nn.Identity()
        self.relu = nn.ReLU() 
        
    def forward(self, x):
        out = self.conv1(x) + self.shortcut(x)
        out = self.relu(out) 
        
        return out 
    
# выход [64, H, W]
class BasicBlock2(nn.Module): 
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64), 
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64)
        ) 
        self.shortcut = nn.Identity()
        self.relu = nn.ReLU() 
        
    def forward(self, x):
        out = self.conv1(x) + self.shortcut(x)
        out = self.relu(out)
        
        return out 
    

# выход [128, H/2, W/2]
class BasicBlock3(nn.Module): 
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128), 
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 1), stride=2, padding=0, bias=False),
            nn.BatchNorm2d(128)
        ) 
        self.relu = nn.ReLU() 
        
    def forward(self, x):
         
        out = self.conv1(x) + self.shortcut(x)
        out = self.relu(out) 
        
        return out 
    
# выход [128, H/2, W/2]
class BasicBlock4(nn.Module): 
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128), 
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.shortcut = nn.Identity()
        self.relu = nn.ReLU() 
        
    def forward(self, x):
         
        out = self.conv1(x) + self.shortcut(x)
        out = self.relu(out) 
        
        return out 
    
    
# выход [256, H/4, W/4]
class BasicBlock5(nn.Module): 
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256), 
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256)
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 1), stride=2, padding=0, bias=False),
            nn.BatchNorm2d(256)
        ) 
        self.relu = nn.ReLU(inplace=True) 
        
    def forward(self, x):
         
        out = self.conv1(x) + self.shortcut(x)
        out = self.relu(out) 
        
        return out 
    
# выход [256, H/4, W/4]
class BasicBlock6(nn.Module): 
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256), 
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256)
        )
        self.shortcut = nn.Identity()
        self.relu = nn.ReLU() 
        
    def forward(self, x):
         
        out = self.conv1(x) + self.shortcut(x)
        out = self.relu(out) 
        
        return out 
    
    
# выход [512, H/8, W/8]
class BasicBlock7(nn.Module): 
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512), 
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512)
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(1, 1), stride=2, padding=0, bias=False),
            nn.BatchNorm2d(512)
        ) 
        self.relu = nn.ReLU(inplace=True) 
        
    def forward(self, x):
         
        out = self.conv1(x) + self.shortcut(x)
        out = self.relu(out) 
        
        return out 
    
# выход [512, H/8, W/8]
class BasicBlock8(nn.Module): 
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512), 
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512)
        )
        self.shortcut = nn.Identity()
        self.relu = nn.ReLU() 
        
    def forward(self, x):
         
        out = self.conv1(x) + self.shortcut(x)
        out = self.relu(out) 
        
        return out 
    
    
class CIFAR_Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
         )
        self.layer1 = nn.Sequential(
            BasicBlock1(), 
            BasicBlock2(),
         )
        self.layer2 = nn.Sequential(
            BasicBlock3(),
            BasicBlock4(),
        )
        
        self.layer3 = nn.Sequential(
            BasicBlock5(),
            BasicBlock6(),
        )

        self.layer4 = nn.Sequential(
            BasicBlock7(),
            BasicBlock8(),
        )
        
        self.linear = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 10, bias=True),
        )
            
    def forward(self, x):
        out = self.conv1(x)
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        out = self.linear(out)
        
        return out   