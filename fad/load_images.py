from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import glob

class ImageDataset(Dataset):
    def __init__(self, root, transform=None, grayscale=False):
        self.root = root
        self.transform = transform
        self.paths = glob.glob(os.path.join(self.root, '**/*.png'), recursive=True)
        self.grayscale = grayscale

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path).convert('L' if self.grayscale else 'RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.paths)

def scale_shift(x):
    return x * 2.0 - 1.0

def load_images(data_path, batch_size, grayscale=False):
    SetRange = transforms.Lambda(scale_shift)
    transform = transforms.Compose([
        transforms.ToTensor(),
        SetRange])

    dataset = ImageDataset(data_path, transform=transform, grayscale=grayscale)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    return dataloader
