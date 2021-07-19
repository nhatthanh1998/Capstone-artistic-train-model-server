from torch.utils.data import DataLoader
from torchvision import datasets
from src.models.extractor import VGG16
from src.models.generator import Generator
from src.utils.util import style_transform, training_transform, gram, mkdir, transform_byte_to_object, \
    save_result, request_save_training_result, check_is_request_deleted, request_start_training
import time
from PIL import Image
import torch
import pika
import datetime
import requests
import event_emitter as events
import socketio

def raiseStop():
    raise StopIteration



class TrainServer:
    def __init__(self, queue_host, main_server_endpoint):
        self.training_request_id = None
        self.main_server_end_point = main_server_endpoint
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.train_dataset = datasets.ImageFolder('src/resources', training_transform)
        self.data_loader = DataLoader(self.train_dataset, batch_size=1)

        self.extractor = VGG16().to(self.device)
        self.criterion = torch.nn.MSELoss().to(self.device)

        self.generator = None
        self.optimizer = None

        self.connection = pika.BlockingConnection(pika.URLParameters(queue_host))
        self.channel = self.connection.channel()

        self.emitter = events.EventEmitter()

        self.sio = socketio.Client()
        self.is_stop = False
        self.sio.connect('ws://localhost:3001')

        @self.sio.on('stop-training')
        def on_message(data):
            print("I'm receive stop-training message")
            self.is_stop = True

    def init_models(self, lr):
        self.generator = Generator().to(self.device)
        self.optimizer = torch.optim.Adam(self.generator.parameters(), lr=lr)

    def calculate_style_grams(self, path):
        style_image = Image.open(requests.get(path, stream=True).raw).convert('RGB')
        style_image = style_transform(style_image).unsqueeze(0).to(self.device)
        B, C, H, W = style_image.shape
        style_features = self.extractor(style_image.expand([1, C, H, W]))
        
        style_gram = {}
        for key, value in style_features.items():
            style_gram[key] = gram(value)
        return style_gram

    def start_training(self, lr, style_photo_path, num_of_iterations, save_step, style_weight, content_weight, training_request_id,
                       relu1_2_weight, relu2_2_weight, relu3_3_weight, relu4_3_weight):
        ts = str(datetime.datetime.now().timestamp())
        output_dir = f"../results/{ts}/outputs"
        snapshot_dir = f"../results/{ts}/snapshots"

        self.init_models(lr)

        style_grams = self.calculate_style_grams(path=style_photo_path)

        # 3. Create output dir and snapshot dir form the training
        mkdir(output_dir)
        mkdir(snapshot_dir)

        style_layer_weight = {
            'relu1_2': relu1_2_weight,
            'relu2_2': relu2_2_weight,
            'relu3_3': relu3_3_weight,
            'relu4_3': relu4_3_weight
        }

        step = 0

        print("start training model")
        for batch_i, (images, _) in enumerate(self.data_loader):
            torch.cuda.empty_cache()
            self.optimizer.zero_grad()
            images_original = images.to(self.device)
            # Generate the output image
            generated_images = self.generator(images_original)
            # Extract features
            original_features = self.extractor(images_original)
            generated_features = self.extractor(generated_images)

            # Calculate content loss
            content_loss = self.criterion(generated_features['relu2_2'], original_features['relu2_2']) * content_weight

            # Calculate style loss
            style_loss = 0
                    
            for key, value in generated_features.items():
                generated_gram = gram(value)
                s_loss = self.criterion(generated_gram, style_grams[key])
                style_loss += s_loss * style_layer_weight[key]

            style_loss = style_loss * style_weight

                    # Calculate total loss
            total_loss = style_loss + content_loss

            # Backward the gradient to the generator node
            total_loss.backward()
            self.optimizer.step()

            # Save training result
            if step % 100 == 0 and step > 0:
                snapshot_s3_path, photo_s3_path = save_result(step, output_dir=output_dir, generator=self.generator,
                                                            result_tensor=generated_images,
                                                            request_id=training_request_id)
                request_save_training_result(request_id=training_request_id, step=step,
                                            server_endpoint=self.main_server_end_point,
                                            snapshot_s3_path=snapshot_s3_path, photo_s3_path=photo_s3_path)
                    
            if step == num_of_iterations: 
                break;

            if self.is_stop == True:
                print("Stop training")
                break;

            step = step + 1

    def process_queue_message(self, ch, method, properties, body):
        # Change is_stop to False before training
        self.is_stop = False
        body = transform_byte_to_object(body)
        lr = body['lr']
        training_request_id = body['id']
        style_photo_path = body['accessURL']
        save_step = body['saveStep']
        content_weight = body['contentWeight']
        style_weight = body['styleWeight']
        relu1_2_weight = body['relu12Weight']
        relu2_2_weight = body['relu22Weight']
        relu3_3_weight = body['relu33Weight']
        relu4_3_weight = body['relu43Weight']
        num_of_iterations = body['numOfIterations']
        isProcess = check_is_request_deleted(training_request_id, self.main_server_end_point)
        if isProcess == True:
            request_start_training(training_request_id, self.main_server_end_point)
            self.training_request_id = training_request_id

            self.start_training(style_photo_path=style_photo_path, num_of_iterations=num_of_iterations, save_step=save_step, lr=lr,
                            style_weight=style_weight, content_weight=content_weight,
                            training_request_id=training_request_id, relu1_2_weight=relu1_2_weight,
                            relu2_2_weight=relu2_2_weight, relu3_3_weight=relu3_3_weight, relu4_3_weight=relu4_3_weight)
        

    def handle_stop_training(self, ch, method, properties, body):
        body = transform_byte_to_object(body)
        print("stop body:", body)
        training_request_id = body['trainingRequestId']
        action = body['action']
        print("EQUAL:", training_request_id == self.training_request_id)
        if action == "STOP" and training_request_id == self.training_request_id:
            print("start event stop event")
            self.training_request_id = None
            self.emitter.emit('stop')

    def init_stop_trainning_queue(self):
        rs = self.channel.queue_declare(queue='', exclusive=True)
        queue_name = rs.method.queue
        self.channel.exchange_declare(exchange="STOP_TRAINING_EXCHANGE", exchange_type='fanout')
        self.channel.queue_bind(exchange="STOP_TRAINING_EXCHANGE", queue=queue_name)
        self.channel.basic_consume(queue=queue_name, on_message_callback=self.handle_stop_training)


    def init_trainning_queue(self):
        self.channel.queue_declare("TRAINING_QUEUE", durable=True)
        self.channel.exchange_declare(exchange="TRAINING_EXCHANGE", exchange_type='direct')
        self.channel.queue_bind(exchange="TRAINING_EXCHANGE", queue="TRAINING_QUEUE", routing_key="")
        self.channel.basic_consume(queue="TRAINING_QUEUE", on_message_callback=self.process_queue_message, auto_ack=True)
        print(f' [*] Waiting for training request. To exit press CTRL+C')

    def start_work(self):
        self.init_stop_trainning_queue()
        self.init_trainning_queue()
        self.channel.start_consuming()
        
