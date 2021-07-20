import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR100, VisionDataset


class CustomCifar100(VisionDataset):
    def __init__(self, root, train, transform=None, target_transform=None):
        super(CustomCifar100, self).__init__(root, transform=transform, target_transform=target_transform)

        self.dataset = CIFAR100(root=root, train=train, download=True, transform=None)
        self.transform = transform
        self.pixel_mean, self.pixel_std = self.get_mean_std(root)

        self.shuffled_classes = self.get_class_order()

        # Create mapping of shuffled classes
        self.label_map = {k: v for v, k in enumerate(self.shuffled_classes)}

        # Define classes per batch starting from shuffled_classes (e.g. class_batches[0]=[3,99,53,21,70,...])
        self.classes_per_batch = self.get_classes_per_batch()

        # Dictionary with key=batch number, value=list
        # The batch number refers to class_batches, while the list contains the indexes of images of the dataset that have a label in that specific batch_class
        self.indexes_per_batch = self.create_batches()

    @staticmethod
    def get_class_order():
        """
        Returns the class order found in ICaRL's code, could be randomized if needed

        Returns:
            list of integers (classes)
        """
        '''class_order: list = [  # Taken from original iCaRL implementation:
            87, 0, 52, 58, 44, 91, 68, 97, 51, 15, 94, 92, 10, 72, 49, 78, 61, 14, 8, 86, 84, 96, 18,
            24, 32, 45, 88, 11, 4, 67, 69, 66, 77, 47, 79, 93, 29, 50, 57, 83, 17, 81, 41, 12, 37, 59,
            25, 20, 80, 73, 1, 28, 6, 46, 62, 82, 53, 9, 31, 75, 38, 63, 33, 74, 27, 22, 36, 3, 16, 21,
            60, 19, 70, 90, 89, 43, 5, 42, 65, 76, 40, 30, 23, 85, 2, 95, 56, 48, 71, 64, 98, 13, 99, 7,
            34, 55, 54, 26, 35, 39
        ]'''
        class_order=[33, 29, 7, 71, 48, 53, 58, 80, 11, 91, 18, 84, 78, 36, 60,
                            1, 96, 90, 57, 54, 85, 17, 4, 92, 51, 99, 24, 95, 88, 89, 47,
                            22, 46, 12, 59, 19, 72, 82, 10, 26, 87, 68, 34, 39, 8, 16, 77,
                            21, 41, 97, 73, 38, 43, 63, 94, 9, 6, 2, 31, 14, 64, 15, 27, 23,
                            37, 45, 49, 74, 65, 83, 40, 75, 62, 50, 61, 79, 69, 81, 25, 66,
                            76, 3, 98, 30, 35, 5, 32, 52, 67, 20, 28, 0, 55, 13, 56, 42, 86,
                            44, 93, 70]
        return class_order

    @staticmethod
    def get_mean_std(root: str):
        """
        Returns the mean and std of the dataset

        Parameters:
            root: path in which teh dataset will be downloaded


        Returns:
            dict: dictionary with each batch as key, each value will contain a list of integers (classes id) 
        """
        dataset: CIFAR100 = CIFAR100(root=root, train=True, transform=transforms.ToTensor())
        loader: DataLoader = DataLoader(dataset, batch_size=1024, num_workers=2)
        channels_sum, channels_sqrd_sum, num_batches = 0, 0, 0
        for images, _ in loader:  # Size([1 or 1024, 3, 32, 32])
            channels_sum += torch.mean(images, dim=[0])
            channels_sqrd_sum += torch.mean(images ** 2, dim=[0])
            num_batches += 1
        mean = channels_sum / num_batches
        # var[X] = E[X**2] - E[X]**2
        std = (channels_sqrd_sum / num_batches - mean ** 2) ** 0.5
        return mean, std

    def get_classes_per_batch(self) -> dict:
        """
        Returns the dict of classes in each batch

        Returns:
            dict: dictionary with each batch as key, each value will contain a list of integers (classes id) 
        """
        classes_in_batches = dict.fromkeys(np.arange(10))  # {0: None, ..., 9: None}
        for i in range(10):
            classes_in_batches[i] = self.shuffled_classes[i*10:(i*10+10)]
        return classes_in_batches

    def create_batches(self) -> dict:
        """
        Returns the batch dict

        Returns:
            batches: dictionary with each batch, each value will contain a list of (img, target) pairs
        """
        batches: dict =  dict((k, []) for k in np.arange(10))
        for idx, item in enumerate(self.dataset):
            for i in range(10):
                if item[1] in self.classes_per_batch[i]:
                    batches[i].append(idx)
        return batches

    def __getitem__(self, index):
        image, label = self.dataset[index]
        if self.transform is not None:
            image = self.transform(image)
            image -= self.pixel_mean  # Here I added don't divide by self.pixel_std, std set to 1
        return image, self.label_map[label]

    def __len__(self):
        return len(self.dataset)


def augmentate(image):
    """
    Returns the input image augmented

    Returns:
        image: tensor with some augmentation performed
    """
    image = F.pad(image, (4, 4, 4, 4), value=0)
    x, y = np.random.randint(8), np.random.randint(8)  # pad=4 => max(x)=8
    image = image[:, x:x+32, y:y+32]
    image = torch.flip(image, [2])
    return image
