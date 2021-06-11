from torch.utils.data import DataLoader
from torchvision import datasets
from src.models.Extractor import Extractor
from src.models.Generator import Generator
from src.utils.util import style_transform, training_transform, gram_matrix
from PIL import Image
import torch
from src.configs.weight import style_layer_weight


class TrainServer:
    def __init__(self, epoch, lr, lambda_style, lambda_content, check_point_after):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.generator = Generator().to(self.device)
        self.extractor = Extractor().to(self.device)
        self.lr = lr
        self.lambda_style = lambda_style
        self.lambda_content = lambda_content
        self.epoch = epoch
        self.check_point_after = check_point_after
        self.style_layer_weight = style_layer_weight

    def process_style_photo(self, path):
        style_image = Image.open(path)
        style_image_ = style_transform(style_image).unsqueeze(0)
        style_features = self.extractor(style_image_.to(self.device))

        style_grams = {}
        for key, value in style_features.items():
            style_grams[key] = gram_matrix(value)

        return style_grams, style_features

    @staticmethod
    def create_data_loader(path):
        train_dataset = datasets.ImageFolder(path, training_transform)
        data_loader = DataLoader(train_dataset, batch_size=4)
        return data_loader

    def start_training(self, style_photo_path):
        # 1. Create DataLoader
        data_loader = self.create_data_loader("./resources/train_images")

        # 2. Process the style photo
        self.process_style_photo(path=style_photo_path)

    def process_queue(self):
        pass
