import torch
from ClassifyFunc.models import CustomResNet
from ClassifyFunc.data_read import get_data
from tqdm import tqdm

from ClassifyFunc.data_preprocess import get_loader_all
from ClassifyFunc.visualization import visualize_predict

# Load data
folder_path = './YinnanTest'
in_channels = 3

samples, labels = get_data(
    folder_path=folder_path,
    in_channels=in_channels,
    # wordIndex=[0,10,11],
    fileIndex=list(range(35,40)),
    # personIndex=[1],
    # txIndex=[0,4,8],
)

print("len(samples): {}".format(len(samples)))
print("len(set(labels)): {}".format(len(set(labels))))

# Create dataset and dataloader
dataloader = get_loader_all(samples, labels)

# Load model
model_path = 'model.pth'  # Path to your saved model
classifier = CustomResNet(in_channels=in_channels,
                          num_classes=len(set(labels)),
                          model='resnet18')
classifier.load_state_dict(torch.load(model_path))
classifier.eval()

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classifier.to(device)

# Prepare for evaluation
all_labels = []
all_preds = []

with torch.no_grad():
    for images, labels in tqdm(dataloader, desc='Processing batches'):
        images, labels = images.to(device), labels.to(device)
        outputs = classifier(images)
        _, predicted = torch.max(outputs.data, 1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())

visualize_predict(all_labels, all_preds)
