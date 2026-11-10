from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

# Channel order inside the generated RGB images.
OMICS_CHANNELS = {"mRNA": 0, "Methylation": 1, "CNV": 2}


class OmicsImageDataset(Dataset):
    """Loads omics images and keeps only the selected channels (for single/dual-omics ablations)."""

    def __init__(self, image_paths, labels, channels=tuple(OMICS_CHANNELS)):
        self.image_paths = image_paths
        self.labels = labels
        self.channel_idx = [OMICS_CHANNELS[name] for name in channels]
        self.to_tensor = ToTensor()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = self.to_tensor(Image.open(self.image_paths[idx]).convert("RGB"))
        return img[self.channel_idx], int(self.labels[idx])
