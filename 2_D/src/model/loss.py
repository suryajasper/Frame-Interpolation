import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models.vgg import vgg16


from src.model.flownet.model import get_flow_net


vgg = None
def get_vgg():
    global vgg
    if vgg is None:
        vgg = vgg16(pretrained=True, progress=False)
        if torch.cuda.is_available():
            vgg.cuda()
        new_classifier = torch.nn.Sequential(*list(vgg.classifier.children())[:-1])
        vgg.classifier = new_classifier
        print(vgg)
    return vgg


def nll_loss(output, target):
    return F.nll_loss(output, target)


def l1_loss(output, target):
    return F.l1_loss(output, target)


def perceptual_loss(predicted_image, true_image):
    vgg = get_vgg()
    features_y = vgg(predicted_image)
    features_x = vgg(true_image)
    return F.mse_loss(features_y, features_x)


def interpolation_loss(predicted_frame, true_frame, beta, gamma):
    loss =  beta * F.l1_loss(predicted_frame, true_frame)
    if gamma != 0:
        loss += gamma * perceptual_loss(predicted_frame, true_frame)
    return loss


def cycle_consistency_loss(predicted_frames, true_frames, alphas, beta, gamma):
    loss = torch.zeros(predicted_frames[0].size(0)).cuda()
    for alpha, predicted_frame, true_frame in zip(alphas, predicted_frames, true_frames):
        loss += alpha * interpolation_loss(predicted_frame, true_frame, beta, gamma)
    return torch.mean(loss)


def motion_linearity_loss(input_frames, predicted_frames):
    frames_0_4 = torch.cat([input_frames[0][None], input_frames[2][None]])
    frames_1_3 = torch.cat([predicted_frames[0][None], predicted_frames[2][None]])

    frames_0_4 = frames_0_4.permute(1, 2, 0, 3, 4)
    frames_1_3 = frames_1_3.permute(1, 2, 0, 3, 4)

    flow_net = get_flow_net()
    flow_0_4 = flow_net(frames_0_4)
    flow_1_3 = flow_net(frames_1_3)

    return F.mse_loss(flow_0_4, 2 * flow_1_3)


def final_loss(input_frames, predicted_frames, true_frames, alphas, beta, gamma, theta):
    alpha_c = cycle_consistency_loss(predicted_frames, true_frames, alphas, beta, gamma)
    alpha_m = motion_linearity_loss(input_frames, predicted_frames)
    return alpha_c + theta * alpha_m
