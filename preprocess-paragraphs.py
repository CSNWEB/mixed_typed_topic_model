#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 11:08:52 2020

@author: christoph
"""

import torch
from torchvision import transforms, models, datasets
import torch.nn as nn
from PIL import Image
import os
import pickle
import gc
import json
from tqdm import tqdm
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )])


class ResNet17(nn.Module):
    def __init__(self, original_model):
        super(ResNet17, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        return x


res18 = models.resnet18(pretrained=True)
res17 = ResNet17(res18)

res17.eval()
torch.set_grad_enabled(False)
print('Reading JSON...')
rows = []
with open('../data/paragraphs_v1.json', mode='r') as json_file:
    data = json.load(json_file)
    line_count = 0
    for row in tqdm(data):
        file_name = f'../data/paragraphs/images/{row["image_id"]}.jpg'
        rows.append({'file_name': file_name, 'image_id': row["image_id"]})
        line_count += 1


print('Processed JSON, transforming images...')

preprocessed = {}
for row in tqdm(rows):
    if os.path.exists(file_name):
        with Image.open(row['file_name']) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img_t = transform(img)
            img.close()
            batch_t = torch.unsqueeze(img_t, 0)
            out = res17(batch_t)
            out = out.detach()
            if row["image_id"] in preprocessed:
                print(row["image_id"])
            else:
                preprocessed[row["image_id"]] = out
print('transformed images, dumping preprocessed:')
pickle.dump(dict(preprocessed), open('preprocessed-paragraphs.pkl', 'wb'))
