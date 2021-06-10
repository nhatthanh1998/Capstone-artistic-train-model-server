from src.models.Extractor import Extractor
from src.models.Generator import Generator
from src.utils.util import style_transform, gram_matrix, create_data_loader
from PIL import Image
import torch


class TrainServer:
    def __init__(self, lr, lambda_style, lambda_content):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.generator = Generator().to(self.device)
        self.extractor = Extractor().to(self.device)
        self.lr = lr
        self.lambda_style = lambda_style
        self.lambda_content = lambda_content

    def process_style_photo(self, path):
        style_image = Image.open(path)
        style_image_ = style_transform(style_image).unsqueeze(0)
        style_features = self.extractor(style_image_.to(self.device))

        style_grams = {}
        for key, value in style_features.items():
            style_grams[key] = gram_matrix(value)

        return style_grams, style_features

    def create_data_loader(self, path):


        pass

    def process_queue(self):
        pass
