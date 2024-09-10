import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

from ClassifyFunc.train import Trainer
from ClassifyFunc.models import CustomResNet
from ClassifyFunc.data_preprocess import get_loader
from ClassifyFunc.data_read import get_data
from ClassifyFunc.visualization import visualize_results

folder_path = './YinnanTest'
in_channels = 3

samples, labels = get_data(
    folder_path=folder_path,
    in_channels=in_channels,
    # wordIndex=[0,10,11],
    fileIndex=list(range(38)),
    # personIndex=[1],
    txIndex=[0,5,9],
)

print("len(samples): {}".format(len(samples)))
print("len(set(labels)): {}".format(len(set(labels))))

trainloader, testloader = get_loader(samples=samples, labels=labels)

# classifier
# classifier = MultiInResNet(num_inputs=NUM_INPUTS,
#                            num_classes=10,
#                            num_in_convs=[1],
#                            in_channels=[3],
#                            out1_channels=[3],
#                            model='resnet18')
classifier = CustomResNet(in_channels=in_channels,
                          num_classes=len(set(labels)),
                          weights=models.ResNet18_Weights.DEFAULT,
                          model='resnet18')

# optimizers
lr = 1e-3
betas = (.5, .99)
optimizer = optim.Adam(classifier.parameters(), lr=lr, betas=betas)
criterion = nn.CrossEntropyLoss()

# train model
NUM_INPUTS = 1
epochs = 30

trainer = Trainer(
    num_inputs=NUM_INPUTS,
    classifier=classifier,
    optimizer=optimizer,
    criterion=criterion,
    print_every=1,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    use_cuda=torch.cuda.is_available(),
    use_scheduler=False)

trainer.train(trainloader=trainloader, testloader=testloader, epochs=epochs)

visualize_results(trainer=trainer)

torch.save(classifier.state_dict(), 'model.pth')
