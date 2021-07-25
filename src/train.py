from torch.utils.data import DataLoader
from torchvision import datasets
from src.models.extractor import VGG16
from src.models.generator import Generator
from src.utils.util import style_transform, training_transform, gram, mkdir, transform_byte_to_object, \
    save_result, request_save_training_result, check_is_request_deleted, request_start_training, \
    request_completed_training
from PIL import Image
import torch
import pika
import datetime
import requests
import socketio
import sys


class TrainServer:
    def __init__(self):
        self.main_server_end_point = "http://backendserverloadbalancer-1655295085.ap-southeast-1.elb.amazonaws.com"
        self.socket_server_end_point = "ws://backendserverloadbalancer-1655295085.ap-southeast-1.elb.amazonaws.com"
        self.connection = None
        self.channel = None
        self.training_request_id = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_dataset = datasets.ImageFolder('src/resources', training_transform)
        self.data_loader = DataLoader(self.train_dataset, batch_size=4)
        self.extractor = VGG16().to(self.device)
        self.criterion = torch.nn.MSELoss().to(self.device)

        self.generator = None
        self.optimizer = None

        self.sio = socketio.Client()
        self.is_stop = False
        self.sio.connect(self.socket_server_end_point)

        @self.sio.on('stop-training')
        def on_message(data):
            print("I'm receive stop-training message")
            self.is_stop = True

    def init_models(self, lr, snapshot_location):
        self.generator = Generator().to(self.device)
        print("Init model")
        if snapshot_location is not None:
            print("load weight")
            self.load_weight(snapshot_location)
        self.optimizer = torch.optim.Adam(self.generator.parameters(), lr=lr)

    def load_weight(self, snapshot_location):
        self.generator.load_state_dict(torch.hub.load_state_dict_from_url(snapshot_location,
                                                                          map_location=torch.device(self.device)))

    def calculate_style_grams(self, path):
        style_image = Image.open(requests.get(path, stream=True).raw).convert('RGB')
        style_image = style_transform(style_image).unsqueeze(0).to(self.device)
        B, C, H, W = style_image.shape
        style_features = self.extractor(style_image.expand([4, C, H, W]))

        style_gram = {}
        for key, value in style_features.items():
            style_gram[key] = gram(value)
        return style_gram

    def start_training(self, lr, style_photo_path, num_of_iterations, save_step, style_weight, content_weight,
                       training_request_id,
                       relu1_2_weight, relu2_2_weight, relu3_3_weight, relu4_3_weight, snapshot_location):
        ts = str(datetime.datetime.now().timestamp())
        output_dir = f"./results/{ts}/outputs"
        snapshot_dir = f"./results/{ts}/snapshots"

        torch.manual_seed(35)
        if self.device == 'cuda':
            torch.cuda.manual_seed(35)

        self.init_models(lr, snapshot_location)

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
        while True:
            if self.is_stop:
                print("Stop training outside")
                break
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
                content_loss = self.criterion(generated_features['relu2_2'],
                                              original_features['relu2_2']) * content_weight

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
                if step % save_step == 0 and step > 0:
                    snapshot_s3_path, photo_s3_path = save_result(step, output_dir=output_dir, generator=self.generator,
                                                                  result_tensor=generated_images,
                                                                  request_id=training_request_id)
                    request_save_training_result(request_id=training_request_id, step=step,
                                                 server_endpoint=self.main_server_end_point,
                                                 snapshot_s3_path=snapshot_s3_path, photo_s3_path=photo_s3_path)

                if step == num_of_iterations:
                    self.is_stop = True
                    request_completed_training(request_id=training_request_id,
                                               main_server_endpoint=self.main_server_end_point)
                    break

                if self.is_stop:
                    print("Stop training")
                    break

                sys.stdout.write(
                    "\r[Step %d] [Total: (%.2f)] [Style Loss: (%.2f)] [Content Loss: (%.2f)]"
                    % (
                        step,
                        total_loss,
                        style_loss,
                        content_loss
                    ))

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
        snapshot_path = body['snapshotLocation']
        isProcess = check_is_request_deleted(training_request_id, self.main_server_end_point)
        if isProcess:
            print("Ack before training")
            ch.basic_ack(delivery_tag=method.delivery_tag)
            request_start_training(training_request_id, self.main_server_end_point)
            self.training_request_id = training_request_id
            self.start_training(style_photo_path=style_photo_path, num_of_iterations=num_of_iterations,
                                save_step=save_step, lr=lr,
                                style_weight=style_weight, content_weight=content_weight,
                                training_request_id=training_request_id, relu1_2_weight=relu1_2_weight,
                                relu2_2_weight=relu2_2_weight, relu3_3_weight=relu3_3_weight,
                                relu4_3_weight=relu4_3_weight, snapshot_location=snapshot_path)
        else:
            ch.basic_ack(delivery_tag=method.delivery_tag)
            print("Ack message")

    def handle_stop_training(self, ch, method, properties, body):
        body = transform_byte_to_object(body)
        training_request_id = body['trainingRequestId']
        action = body['action']
        if action == "STOP" and training_request_id == self.training_request_id:
            print("start event stop event")
            self.training_request_id = None

    def init_stop_training_queue(self):
        rs = self.channel.queue_declare(queue='', exclusive=True)
        queue_name = rs.method.queue
        self.channel.exchange_declare(exchange="STOP_TRAINING_EXCHANGE", exchange_type='fanout')
        self.channel.queue_bind(exchange="STOP_TRAINING_EXCHANGE", queue=queue_name)
        self.channel.basic_consume(queue=queue_name, on_message_callback=self.handle_stop_training)

    def init_training_queue(self):
        self.channel.queue_declare("TRAINING_QUEUE", durable=True)
        self.channel.exchange_declare(exchange="TRAINING_EXCHANGE", exchange_type='direct')
        self.channel.queue_bind(exchange="TRAINING_EXCHANGE", queue="TRAINING_QUEUE", routing_key="")
        self.channel.basic_consume(queue="TRAINING_QUEUE", on_message_callback=self.process_queue_message,
                                   auto_ack=False)
        print(f' [*] Waiting for training request. To exit press CTRL+C')

    def start_work(self):
        while True:
            try:
                print("Connecting...")
                self.connection = pika.BlockingConnection(pika.URLParameters(
                    "amqps://nhatthanhlolo1:nhatthanh123@b-bb75efcd-b132-429f-9d91-9a062463a388.mq.ap-southeast-1.amazonaws.com:5671"))
                self.channel = self.connection.channel()
                self.init_stop_training_queue()
                self.init_training_queue()
                self.channel.basic_qos(prefetch_count=1)
                self.channel.start_consuming()
            except KeyboardInterrupt:
                self.channel.stop_consuming()
                self.connection.close()
                break
            except pika.exceptions.ConnectionClosedByBroker:
                continue
            except pika.exceptions.AMQPChannelError as err:
                break
            except pika.exceptions.AMQPConnectionError:
                print("Connection was closed, retrying...")
                continue
