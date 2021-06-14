from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.utils import save_image
from src.models.Extractor import Extractor
from src.models.Generator import Generator
from src.utils.util import style_transform, training_transform, gram_matrix, save_model, mkdir
from PIL import Image
import torch
from src.configs.weight import style_layer_weight
import sys
import pika


class TrainServer:
    def __init__(self, epoch, lr, lambda_style, lambda_content, queue_host, exchange_train_server_name, routing_key):
        self.routing_key = routing_key
        self.queue_host = queue_host
        self.exchange_train_server_name = exchange_train_server_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lr = lr
        self.lambda_style = lambda_style
        self.lambda_content = lambda_content
        self.epoch = epoch
        self.style_layer_weight = style_layer_weight
        self.generator = Generator().to(self.device)
        self.extractor = Extractor().to(self.device)
        self.optimizer = torch.optim.Adam(self.generator.parameters(), lr=self.lr)
        self.criterion = torch.nn.MSELoss().to(self.device)
        self.connection = pika.BlockingConnection(pika.URLParameters(self.queue_host))
        self.channel = self.connection.channel()

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
        return data_loader, train_dataset

    def start_training(self, style_photo_path, style_name, check_point_after):
        output_dir = f"../output/{style_name}"
        snapshot_dir = f"../snapshot/{style_name}"

        # 1. Create DataLoader
        data_loader, train_dataset = self.create_data_loader("./resources/train_images")

        # 2. Process the style photo
        style_grams, style_features = self.process_style_photo(path=style_photo_path)

        # 3. Create output dir and snapshot dir form the training
        mkdir(output_dir)
        mkdir(snapshot_dir)

        # 4. start training
        total_batch = (len(train_dataset) / 4) - 1

        for i in range(self.epoch):
            for batch_i, (images, _) in enumerate(data_loader):
                torch.cuda.empty_cache()
                self.optimizer.zero_grad()
                images_original = images.to(self.device)

                # Generate the output image
                generated_images = self.generator(images_original)

                # Extract features
                original_features = self.extractor(images_original)
                generated_features = self.extractor(generated_images)

                # Calculate content loss
                content_loss = self.criterion(generated_features['relu2_2'], original_features['relu2_2']) * self.lambda_content

                # Calculate style loss
                style_loss = 0
                for key, value in generated_features.items():
                    generated_gram = gram_matrix(value)
                    s_loss = self.criterion(generated_gram, style_grams[key])
                    style_loss += s_loss * style_layer_weight[key]
                style_loss = style_loss * self.lambda_style

                # Calculate total loss
                total_loss = style_loss + content_loss

                # Backward the gradient to the generator node
                total_loss.backward()
                self.optimizer.step()

                # Update batch done
                batches_done = i * len(data_loader) + batch_i
                if batches_done % 500 == 0:
                    save_image(generated_images, f"{output_dir}/step_{batches_done}.jpg")
                sys.stdout.write(
                    "\r[Epoch %d/%d] [Step %d] [Batch %d/%d] [Total: (%.2f)] [Style Loss: (%.2f)] [Content Loss: (%.2f)]"
                    % (
                        i + 1,
                        self.epoch,
                        batches_done,
                        batch_i,
                        total_batch,
                        total_loss,
                        style_loss,
                        content_loss
                    )
                )

                if batches_done % self.check_point_after == 0 and batches_done > 1:
                    save_model(batches_done)

    def process_queue_message(self, ch, method, properties, body):
        pass

    def start_work(self):
        self.channel.queue_declare(self.routing_key, durable=True)
        self.channel.exchange_declare(exchange=self.exchange_train_server_name, exchange_type='direct')
        self.channel.queue_bind(exchange=self.exchange_train_server_name, queue=self.routing_key,
                                routing_key=self.routing_key)
        self.channel.basic_consume(queue=self.routing_key, on_message_callback=self.process_queue_message)
        print(f' [*] Waiting for messages at exchange {self.exchange_train_server_name} routing Key: {self.routing_key}. To exit press CTRL+C')
        print(
            f' [*] Waiting for messages at exchange. To exit press CTRL+C')
