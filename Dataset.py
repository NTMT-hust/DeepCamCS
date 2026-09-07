from PIL import Image
from torch.utils.data import Dataset

class ImbalancedImageDataset(Dataset):
    def __init__(self, image_paths, labels, class_counts,
                transform=None, use_class_aware_aug=False):
        self.image_paths = image_paths
        self.labels = labels
        self.class_counts = class_counts
        self.transform = transform
        self.use_class_aware_aug = use_class_aware_aug

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label