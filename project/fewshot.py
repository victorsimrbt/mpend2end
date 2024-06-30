import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18
from tqdm import tqdm
from matplotlib import pyplot as plt
from PIL import Image
import torch
import torchvision.transforms as transforms
import os


class prototype(nn.Module):
    def __init__(self, backbone: nn.Module):
        super(prototype, self).__init__()
        self.backbone = backbone

    def forward(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict query labels using labeled support images.
        """
        z_support = self.backbone.forward(support_images)
        z_query = self.backbone.forward(query_images)
        # ! encode support and query images

        n_way = len(torch.unique(support_labels))
        z_proto = torch.cat(
            [
                z_support[torch.nonzero(support_labels == label)].mean(0)
                for label in range(n_way)
            ]
        )
        # ! calculathe prototypes for both

        dists = torch.cdist(z_query, z_proto)
        scores = -dists
        return scores


convolutional_network = resnet18(pretrained=True)
convolutional_network.fc = nn.Flatten()
print(convolutional_network)

model = prototype(convolutional_network).cuda()
image_size = 256

def get_img(path):
    image = Image.open(path)
    totensor = transforms.ToTensor()
    transform=transforms.Compose(
        [
            transforms.Resize([int(image_size * 1.15), int(image_size * 1.15)]),
            transforms.CenterCrop(image_size),
        ]
    )
    # ! transforms iamge to fit into feature matching network
    transformed = transform(image)
    return totensor(transformed)[:3]

def get_imgs(path):
    tensor = []
    imgs = os.listdir(path)
    for img in imgs:
        img_path = os.path.join(path,img)
        tensor.append(get_img(img_path))
    # ! get images and convert to one tensor for input
    return torch.stack(tensor)

def train_test(tensor,split=0.8):
    idx = int(tensor.shape[0] * split)
    return tensor[:idx],tensor[idx:]

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

def fit(
    support_images: torch.Tensor,
    support_labels: torch.Tensor,
    query_images: torch.Tensor,
    query_labels: torch.Tensor,
) -> float:
    optimizer.zero_grad()
    classification_scores = model(
        support_images.cuda(), support_labels.cuda(), query_images.cuda()
    )

    loss = criterion(classification_scores, query_labels.cuda())
    loss.backward()
    optimizer.step()
    # ! minimise the classification error for query images
    return loss.item()

def construct(lsts):
    imgs = torch.concatenate(lsts)
    labels = []
    cnt = 0 
    for val in lsts:
        labels += [cnt for i in range(len(val))]
        cnt += 1
    labels = torch.Tensor(labels)
    # ! create image, label pairs
    return imgs,labels
    


BASE = r"..\fewshot\mpdata"
bead = get_imgs(os.path.join(BASE,"Bead")).cuda()
fiber = get_imgs(os.path.join(BASE,"Fiber")).cuda()
fragment = get_imgs(os.path.join(BASE,"Fragment")).cuda()
negative = get_imgs(r"..\fewshot\mpdata").cuda()

bead_train,bead_test = train_test(bead)
fiber_train,fiber_test = train_test(fiber)
fragment_train,fragment_test = train_test(fragment)
negative_train,negative_test = train_test(negative)

labels = []

test_images,test_labels = construct([negative_test,bead_test,fiber_test,fragment_test])
train_images,train_labels = construct([negative_train,bead_train,fiber_train,fragment_train])

scores = model(
    train_images.cuda(),
    train_labels.cuda(),
    test_images.cuda()
)

_, pred_labels = torch.max(scores.data, 1)

print("Ground Truth / Predicted")
total = 0
for i in range(len(pred_labels)):
    if pred_labels[i] == test_labels[i]:
        total += 1
    # print(pred_labels[i],test_labels[i]
print(total/len(pred_labels))
