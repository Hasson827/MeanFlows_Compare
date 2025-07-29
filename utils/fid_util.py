import torch
import torchvision
import numpy as np
import torch.nn.functional as F
import os
from torch.utils.data import DataLoader

def get_real_images(dataset_type, image_size, batch_size, num_images=10000):
    if dataset_type == 'CIFAR10':
        dataset = torchvision.datasets.CIFAR10(
            root='data', train=False, download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.Resize(image_size),
                torchvision.transforms.ToTensor(),
            ])
        )
    elif dataset_type == 'MNIST':
        dataset = torchvision.datasets.MNIST(
            root='data', train=False, download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.Resize(image_size),
                torchvision.transforms.ToTensor(),
            ])
        )
    else:
        raise ValueError('Unsupported dataset type')
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    images = []
    for x, _ in loader:
        images.append(x)
        if len(images) * batch_size >= num_images:
            break
    images = torch.cat(images, dim=0)[:num_images]
    return images

def get_inception_model(device='cuda'):
    weights = torchvision.models.Inception_V3_Weights.DEFAULT
    model = torchvision.models.inception_v3(weights=weights, transform_input=False)
    model.fc = torch.nn.Identity()
    model.eval().to(device)
    return model

def get_features(images, model, device='cuda', batch_size=64):
    features = []
    n = images.shape[0]
    for i in range(0, n, batch_size):
        batch = images[i:i+batch_size].to(device)
        if batch.min() < 0:
            batch = (batch + 1) / 2
        batch = batch.clamp(0, 1)
        if batch.shape[1] == 1:
            batch = batch.repeat(1, 3, 1, 1)
        batch = F.interpolate(batch, size=(299, 299), mode='bilinear', align_corners=False)
        with torch.no_grad():
            feat = model(batch)
        features.append(feat.cpu())
    return torch.cat(features, dim=0).numpy()

def calculate_fid_from_features(mu1, sigma1, mu2, sigma2, eps=1e-6):
    from scipy.linalg import sqrtm
    covmean = sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = np.sum((mu1 - mu2) ** 2) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(fid)

def calculate_fid(real_images=None, fake_images=None, device='cuda'):
    inception = get_inception_model(device)
    real_feats = get_features(real_images, inception, device)
    mu_real, sigma_real = real_feats.mean(0), np.cov(real_feats, rowvar=False)
    assert fake_images is not None, 'fake_images must be provided.'
    inception = get_inception_model(device)
    fake_feats = get_features(fake_images, inception, device)
    mu_fake, sigma_fake = fake_feats.mean(0), np.cov(fake_feats, rowvar=False)
    fid_score = calculate_fid_from_features(mu_real, sigma_real, mu_fake, sigma_fake)
    return fid_score
