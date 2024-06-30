import os
from pegbis.main import segment
from PIL import Image
import imageio
import cv2
import torchvision.transforms as transforms
from fewshot import prototype
import torch
from torch import nn, optim
from torchvision import transforms
from torchvision.models import resnet18
import torchvision.transforms as transforms
import numpy as np

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
    transformed = transform(image)
    return totensor(transformed)[:3]


def get_imgs(path):
    tensor = []
    imgs = os.listdir(path)
    for img in imgs:
        img_path = os.path.join(path,img)
        tensor.append(get_img(img_path))
    return torch.stack(tensor)

def train_test(tensor,split=0.8):
    idx = int(tensor.shape[0] * split)
    return tensor[:idx],tensor[idx:]

def construct(lsts):
    imgs = torch.concatenate(lsts)
    labels = []
    cnt = 0 
    for val in lsts:
        labels += [cnt for i in range(len(val))]
        cnt += 1
    labels = torch.Tensor(labels)
    return imgs,labels

BASE = "..\captures"
imgs = [os.path.join(BASE,img) for img in os.listdir(BASE)]


save = "..\captures"

cnt = 0

convolutional_network = resnet18(pretrained=True)
convolutional_network.fc = nn.Flatten()
print(convolutional_network)

model = prototype(convolutional_network).cuda()

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

for i in range(len(imgs)):
    image = imageio.imread(os.path.join(BASE,imgs[i]))
    height, width = image.shape[:2]
    w = width//4
    h = height//4
    img = cv2.resize(image, (w,h))
    output,u = segment(img,0.5,300,100)

    vals = {}
    # ! vals[x] = [min_x,max_x,min_y,max_y]

    for y in range(h):
        for x in range(w):
            obj = u.find(y * w + x)
            if obj in vals:
                min_x,max_x,min_y,max_y = vals[obj]
                vals[obj] = [min(x,min_x),max(x,max_x),min(y,min_y),max(y,max_y)]
            else:
                vals[obj] = [x,y,x,y]

    cnt += 1
    file_name = "image_{}.jpg".format(cnt)
    
    new = np.zeros(img.shape)
    for obj in vals:
        min_x,max_x,min_y,max_y = vals[obj]
        if max_x-min_x > 50 and max_y-min_y > 50:
            obj_img = img[min_y:max_y+1, min_x:max_x+1]
            pred = model(train_images,train_labels,obj_img.reshape(1,obj.shape[0],obj.shape[1]))
            if pred != 0:
                new[min_y:max_y+1, min_x:max_x+1] = obj_img
                # ! if the classification model determines that it is a microplastic, save it
                
        image = Image.fromarray(new)

    # Save the image to a file
    image.save(r"..\captures\file_name")

              
            # plt.imshow(obj_img)
            # plt.show()
        