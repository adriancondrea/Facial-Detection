import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from constants import IMAGE_SIZE, IMAGE_CLASSES, TRAINING_PERCENTAGE

device = torch.device('cuda:0' if torch.cuda.is_available() else
                      'cpu')


class ImageClassifierDataset(Dataset):
    def __init__(self, image_classes=IMAGE_CLASSES):
        self.images = []
        self.labels = []
        self.classes = list(set(image_classes))
        self.class_to_label = {c: float(i) for i, c in
                               enumerate(self.classes)}
        self.image_size = IMAGE_SIZE
        self.transforms = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def load_data(self):
        print('loading face images...')
        self.load_from_directory('images/faces', 'face')
        print('loading not face images...')
        for subdirectory in os.listdir('images/not_faces'):
            if not subdirectory.startswith('.'):
                print(f'loading from directory {subdirectory}')
                self.load_from_directory(f'images/not_faces/{subdirectory}', 'not face')

    def process_images(self, image_list, image_classes):
        for image, image_class in zip(image_list, image_classes):
            transformed_image = self.transforms(image)
            self.images.append(transformed_image)
            label = self.class_to_label[image_class]
            self.labels.append(label)

    def load_from_directory(self, directory_name, label):
        images = []
        directory_images = [file_name for file_name in os.listdir(directory_name) if not file_name.startswith('.')]
        np.random.shuffle(directory_images)

        for image in directory_images:
            images.append(Image.open(f"{directory_name}/{image}").convert('RGB'))

        self.process_images(images, [label for _ in images])

    def split(self):
        images_labels = list(zip(self.images, self.labels))
        np.random.shuffle(images_labels)
        indexes = list(range(len(images_labels)))
        train_indexes = []
        validation_indexes = []
        for index in indexes:
            if random.random() < TRAINING_PERCENTAGE:
                train_indexes.append(index)
            else:
                validation_indexes.append(index)

        trainSet = ImageClassifierDataset()
        trainSet.images = [images_labels[i][0] for i in train_indexes]
        trainSet.labels = [torch.tensor([images_labels[i][1]]) for i in train_indexes]

        trainSet.images = torch.stack(trainSet.images)
        trainSet.labels = torch.stack(trainSet.labels)

        testSet = ImageClassifierDataset()
        testSet.images = [images_labels[i][0] for i in validation_indexes]
        testSet.labels = [torch.tensor([images_labels[i][1]]) for i in validation_indexes]

        testSet.images = torch.stack(testSet.images)
        testSet.labels = torch.stack(testSet.labels)

        return trainSet, testSet

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return len(self.images)

# dataset = ImageClassifierDataset()
# dataset.load_data()
# print(len(dataset.images))
# dataset.split()
