import boto3
from dotenv import load_dotenv
from torchvision import transforms as transforms
import torch
import os
import json
import requests

from torchvision.utils import save_image

load_dotenv()
ENV = os.environ.get("ENV", "dev")
S3_BUCKET = 'artisan-photos'


def init_s3_bucket(env, bucket):
    if env == "production":
        s3_client = boto3.client('s3')
    else:
        AWS_PUBLIC_KEY = os.environ.get("AWS_PUBLIC_KEY")
        AWS_PRIVATE_KEY = os.environ.get("AWS_PRIVATE_KEY")
        session = boto3.Session(
            aws_access_key_id=AWS_PUBLIC_KEY,
            aws_secret_access_key=AWS_PRIVATE_KEY
        )
        s3_client = session.client('s3')

    region = s3_client.get_bucket_location(Bucket=bucket)['LocationConstraint']

    return s3_client, region


s3, S3_REGION = init_s3_bucket(bucket=S3_BUCKET, env=ENV)


def save_file_to_s3(file_path, folder_name):
    file_name = file_path[file_path.rindex('/') + 1 : ]
    save_s3_path = f"training-result/{folder_name}/{file_name}"
    s3.upload_file(file_path, S3_BUCKET, save_s3_path)
    return save_s3_path


""" Transforms for training images """
training_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)


""" Transforms for training images """
style_transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)


def gram(tensor):
    B, C, H, W = tensor.shape
    x = tensor.view(B, C, H*W)
    x_t = x.transpose(1, 2)
    return  torch.bmm(x, x_t) / (C*H*W)


def save_result(i, output_dir, generator, result_tensor, request_id):
    save_snapshot_path = f'{output_dir}/generator_{str(i)}.pth'
    save_image_path = f'{output_dir}/result_{str(i)}.png'
    save_image(result_tensor, save_image_path)
    torch.save(generator.state_dict(), save_snapshot_path)
    snapshot_s3_path = save_file_to_s3(save_snapshot_path, request_id)
    photo_s3_path = save_file_to_s3(save_image_path, request_id)
    return snapshot_s3_path, photo_s3_path


def request_save_training_result(request_id, step, snapshot_s3_path, photo_s3_path, server_endpoint):
    payload = {
        "trainingRequestId": request_id,
        "step": step,
        "snapshotLocation": snapshot_s3_path,
        "resultPhotoLocation": photo_s3_path
    }
    requests.post(f'{server_endpoint}/training-results', payload)


def mkdir(path):
    print("mkdir.......")
    os.makedirs(path, exist_ok=True)  # succeeds even if directory exists.


def transform_byte_to_object(byte_data):
    response = byte_data.decode('utf8')
    response = json.loads(response)
    return response


