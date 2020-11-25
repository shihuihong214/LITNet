import numpy as np
import torch
import os
import argparse
import time
import itertools

from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

import utils
from network import ImageTransformNet, ImageTransformNet_dpws, distiller_1, distiller_2
from vgg import Vgg16

# Global Variables
IMAGE_SIZE = 256
BATCH_SIZE = 4
LEARNING_RATE = 1e-3
EPOCHS = 2

STYLE_WEIGHT = 5e4
CONTENT_WEIGHT = 1e-2
L1_WEIGHT = 1
DIS_WEIGHT = 1
TV_WEIGHT = 0
# STYLE_WEIGHT = 0
# CONTENT_WEIGHT = 0
# L1_WEIGHT = 1
# DIS_WEIGHT = 0
# TV_WEIGHT = 0

def train(args):
    # GPU enabling
    if (args.gpu != None):
        use_cuda = True
        dtype = torch.cuda.FloatTensor
        torch.cuda.set_device(args.gpu)
        print("Current device: %d" %torch.cuda.current_device())

    # visualization of training controlled by flag
    visualize = (args.visualize != None)
    if (visualize):
        img_transform_512 = transforms.Compose([
            transforms.Scale(512),                  # scale shortest side to image_size
            # transforms.CenterCrop(512),             # crop center image_size out
            transforms.ToTensor(),                  # turn image from [0-255] to [0-1]
            utils.normalize_tensor_transform()      # normalize with ImageNet values
        ])

        # testImage_amber = utils.load_image("content_imgs/amber.jpg")
        # testImage_amber = img_transform_512(testImage_amber)
        # testImage_amber = Variable(testImage_amber.repeat(1, 1, 1, 1), requires_grad=False).type(dtype)

        # testImage_dan = utils.load_image("content_imgs/dan.jpg")
        # testImage_dan = img_transform_512(testImage_dan)
        # testImage_dan = Variable(testImage_dan.repeat(1, 1, 1, 1), requires_grad=False).type(dtype)

        testImage_maine = utils.load_image("content_imgs/chicago.jpg")
        testImage_maine = img_transform_512(testImage_maine)
        testImage_maine = Variable(testImage_maine.repeat(1, 1, 1, 1), requires_grad=False).type(dtype)

    # define network
    image_transformer_dpws = ImageTransformNet_dpws().type(dtype)
    paras = [image_transformer_dpws.parameters()]
    # optimizer = Adam(image_transformer_dpws.parameters(), LEARNING_RATE) 

    loss_mse = torch.nn.MSELoss()
    loss_l1 = torch.nn.L1Loss()

    # load vgg network
    dis_1 = distiller_1().type(dtype)
    dis_2 = distiller_2().type(dtype)
    paras.append(dis_1.parameters())
    paras.append(dis_2.parameters())
    optimizer = Adam(itertools.chain(*paras), LEARNING_RATE) 
   
    vgg = Vgg16().type(dtype)
    image_transformer = ImageTransformNet().type(dtype)
    image_transformer.load_state_dict(torch.load('models/starry_night_crop_nosm.model'))

    # get training dataset
    dataset_transform = transforms.Compose([
        transforms.Scale(IMAGE_SIZE),           # scale shortest side to image_size
        transforms.CenterCrop(IMAGE_SIZE),      # crop center image_size out
        transforms.ToTensor(),                  # turn image from [0-255] to [0-1]
        utils.normalize_tensor_transform()      # normalize with ImageNet values
    ])
    train_dataset = datasets.ImageFolder(args.dataset, dataset_transform)
    train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE)

    # style image
    style_transform = transforms.Compose([
        transforms.ToTensor(),                  # turn image from [0-255] to [0-1]
        utils.normalize_tensor_transform()      # normalize with ImageNet values
    ])
    style = utils.load_image(args.style_image)
    style = style_transform(style)
    style = Variable(style.repeat(BATCH_SIZE, 1, 1, 1)).type(dtype)
    style_name = os.path.split(args.style_image)[-1].split('.')[0]
    
    # calculate gram matrices for style feature layer maps we care about
    style_features = vgg(style)
    style_gram = [utils.gram(fmap) for fmap in style_features]

    for e in range(EPOCHS):

        # track values for...
        img_count = 0
        aggregate_style_loss = 0.0
        aggregate_content_loss = 0.0
        aggregate_l1_loss = 0.0
        aggregate_tv_loss = 0.0

        # train network
        image_transformer_dpws.train()
        for batch_num, (x, label) in enumerate(train_loader):
            img_batch_read = len(x)
            img_count += img_batch_read

            # zero out gradients
            optimizer.zero_grad()

            # input batch to transformer network

            x = Variable(x).type(dtype)
            y_hat = image_transformer_dpws(x)
            y_label = image_transformer(x)

            # ###### distiller ###########
            dis1 = dis_1(y_label[1])
            dis2 = dis_2(y_label[2])
            dis_loss = loss_mse(dis1, y_hat[1])
            dis_loss += loss_mse(dis2, y_hat[2])
            dis_loss = DIS_WEIGHT*dis_loss
            # aggregate_l1_loss += l1_loss.data.item()

            # get vgg features
            y_c_features = vgg(y_label[0])
            y_hat_features = vgg(y_hat[0])

            # calculate style loss
            y_hat_gram = [utils.gram(fmap) for fmap in y_hat_features]
            style_loss = 0.0
            # TODO:
            # style_weight = [5, 1, 1, 1]
            style_weight = [1, 1, 1, 1]
            for j in range(4):
                style_loss += loss_mse(y_hat_gram[j], style_gram[j][:img_batch_read])
                style_loss = style_weight[j]*style_loss
            style_loss = STYLE_WEIGHT*style_loss
            aggregate_style_loss += style_loss.data.item()

            # calculate content loss (h_relu_2_2)
            recon = y_c_features[1]      
            recon_hat = y_hat_features[1]
            content_loss = CONTENT_WEIGHT*loss_mse(recon_hat, recon)
            aggregate_content_loss += content_loss.data.item()

            # calculate l1 loss
            l1_loss = L1_WEIGHT*loss_mse(y_hat[0], y_label[0])
            aggregate_l1_loss += l1_loss.data.item()

            # calculate total variation regularization (anisotropic version)
            # https://www.wikiwand.com/en/Total_variation_denoising
            diff_i = torch.sum(torch.abs(y_hat[0][:, :, :, 1:] - y_hat[0][:, :, :, :-1]))
            diff_j = torch.sum(torch.abs(y_hat[0][:, :, 1:, :] - y_hat[0][:, :, :-1, :]))
            tv_loss = TV_WEIGHT*(diff_i + diff_j)
            aggregate_tv_loss += tv_loss.data.item()

            # total loss
            total_loss = style_loss + content_loss + tv_loss + l1_loss + dis_loss

            # backprop
            total_loss.backward()
            optimizer.step()

            # print out status message
            if ((batch_num + 1) % 100 == 0):
                status = "{}  Epoch {}:  [{}/{}]  Batch:[{}]  agg_style: {:.6f}  agg_content: {:.6f}  agg_l1: {:.6f} ".format(
                                time.ctime(), e + 1, img_count, len(train_dataset), batch_num+1,
                                aggregate_style_loss/(batch_num+1.0), aggregate_content_loss/(batch_num+1.0), aggregate_l1_loss/(batch_num+1.0),
                                )
                print(status)

            if ((batch_num + 1) % 5000 == 0) and (visualize):
                image_transformer_dpws.eval()

                if not os.path.exists("visualization"):
                    os.makedirs("visualization")
                if not os.path.exists("visualization/%s" %style_name):
                    os.makedirs("visualization/%s" %style_name)

                # outputTestImage_amber = image_transformer(testImage_amber).cpu()
                # amber_path = "visualization/%s/amber_%d_%05d.jpg" %(style_name, e+1, batch_num+1)
                # utils.save_image(amber_path, outputTestImage_amber.data[0])

                # outputTestImage_dan = image_transformer(testImage_dan).cpu()
                # dan_path = "visualization/%s/dan_%d_%05d.jpg" %(style_name, e+1, batch_num+1)
                # utils.save_image(dan_path, outputTestImage_dan.data[0])

                outputTestImage_maine = image_transformer_dpws(testImage_maine)
                # outputTestImage_maine[0] = outputTestImage_maine[0].cpu()
                maine_path = "visualization/%s/chicago%d_%05d.jpg" %(style_name, e+1, batch_num+1)
                utils.save_image(maine_path, outputTestImage_maine[0].data[0].cpu())

                print("images saved")
                image_transformer_dpws.train()

    # save model
    image_transformer_dpws.eval()

    if use_cuda:
        image_transformer_dpws.cpu()

    if not os.path.exists("models"):
        os.makedirs("models")
    filename = "models/smallest.model"
    torch.save(image_transformer_dpws.state_dict(), filename)
    
    if use_cuda:
        image_transformer_dpws.cuda()

def style_transfer(args):
    # GPU enabling
    if (args.gpu != None):
        use_cuda = True
        dtype = torch.cuda.FloatTensor
        torch.cuda.set_device(args.gpu)
        print("Current device: %d" %torch.cuda.current_device())

    # content image
    img_transform_512 = transforms.Compose([
            transforms.Scale(512),                  # scale shortest side to image_size
            # transforms.CenterCrop(512),             # crop center image_size out
            transforms.ToTensor(),                  # turn image from [0-255] to [0-1]
            utils.normalize_tensor_transform()      # normalize with ImageNet values
    ])

    content = utils.load_image(args.source)
    content = img_transform_512(content)
    content = content.unsqueeze(0)
    content = Variable(content).type(dtype)

    # load style model
    style_model = ImageTransformNet().type(dtype)
    style_model.load_state_dict(torch.load(args.model_path))

    # process input image
    stylized = style_model(content)
    utils.save_image(args.output, stylized[0].data[0].cpu())


def main():
    parser = argparse.ArgumentParser(description='style transfer in pytorch')
    subparsers = parser.add_subparsers(title="subcommands", dest="subcommand")

    train_parser = subparsers.add_parser("train", help="train a model to do style transfer")
    train_parser.add_argument("--style-image", type=str, required=True, help="path to a style image to train with")
    train_parser.add_argument("--dataset", type=str, required=True, help="path to a dataset")
    train_parser.add_argument("--gpu", type=int, default=None, help="ID of GPU to be used")
    train_parser.add_argument("--visualize", type=int, default=None, help="Set to 1 if you want to visualize training")

    style_parser = subparsers.add_parser("transfer", help="do style transfer with a trained model")
    style_parser.add_argument("--model-path", type=str, required=True, help="path to a pretrained model for a style image")
    style_parser.add_argument("--source", type=str, required=True, help="path to source image")
    style_parser.add_argument("--output", type=str, required=True, help="file name for stylized output image")
    style_parser.add_argument("--gpu", type=int, default=None, help="ID of GPU to be used")

    args = parser.parse_args()

    # command
    if (args.subcommand == "train"):
        print("Training!")
        train(args)
    elif (args.subcommand == "transfer"):
        print("Style transfering!")
        style_transfer(args)
    else:
        print("invalid command")

if __name__ == '__main__':
    main()








