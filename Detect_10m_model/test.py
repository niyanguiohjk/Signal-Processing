import torch
import torch.nn as nn
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(5000, 2048),                                                           
            nn.LeakyReLU(0.2, inplace=True), 
            nn.Dropout(0.2),
            # nn.Linear(2048, 1024),                        
            # nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout(0.2),  
            # nn.Linear(1024, 512),                        
            # nn.LeakyReLU(0.2, inplace=True), 
            # nn.Dropout(0.2), 
            nn.Linear(2048, 512),                        
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),  
            nn.Linear(512, 256),                        
            nn.LeakyReLU(0.2, inplace=True),  
            nn.Dropout(0.2),       
            nn.Linear(256, 1)                                                      
        )

    def forward(self, x):
        validity = self.model(x)                      
        return validity  
# model = Discriminator()
model = torch.load('./GAN/discriminator.pth')
model = nn.Sequential(*list(model.model.children())[:-2])
print(model())