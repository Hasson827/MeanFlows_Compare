import os
import time
import torch
import numpy as np
from accelerate import Accelerator
from torchvision.utils import make_grid, save_image

from models.dit import MFDiT
from models.meanflows import MeanFlow, AdditiveMeanFlow
from utils.data_util import create_dataloader
from utils.ema_util import EMA
from utils.fid_util import calculate_fid, get_real_images
from utils.vis_util import plot_training_curves, write_training_report

import warnings
warnings.filterwarnings("ignore", message="No device id is provided via `init_process_group` or `barrier `.")

DATA_CONFIG = dict(
    dataset_type = 'MNIST',
    image_size = 28,
    batch_size = 64,
    num_workers = 8
)

EXPERIMENT_CONFIG = dict(
    num_epochs = 1000,
    warmup_epoch = 50,
    learning_rate = 6e-4,
    weight_decay = 0,
    betas = (0.9, 0.999),
    mixed_precision = 'bf16',
    log_epoch = 5, 
    sample_epoch = 25, 
    use_cond = True
)

DIT_CONFIG = dict(
    input_size = DATA_CONFIG['image_size'],
    patch_size = 2,
    in_channels = 1,
    dim = 512,
    depth = 8,
    num_heads = 8,
    mlp_ratio = 4.0,
    num_classes = 10,
    dropout = 0.2
)

MEANFLOW_CONFIG = dict(
    channels = DIT_CONFIG['in_channels'],
    image_size = DATA_CONFIG['image_size'],
    num_classes = DIT_CONFIG['num_classes'],
    normalizer = ['minmax', None, None],
    flow_ratio = 0.25,
    time_dist = ['lognorm', -2.0, 2.0],
    cfg_ratio = 0.1,
    omega = 1.0,
    kappa = 0.5
)

ALPHA_CONFIG = dict(
    final_alpha = 0.5, 
    final_epoch = 0.9 * EXPERIMENT_CONFIG['num_epochs']
)

def get_lr(epoch, config):
    warmup_epoch = config['warmup_epoch']
    final_epoch = config['num_epochs']
    lr = config['learning_rate']
    if epoch < warmup_epoch:
        return lr * ((epoch + 1) / warmup_epoch)
    progress = (epoch - warmup_epoch) / max(1, final_epoch - warmup_epoch)
    cosine = 0.5 * (1 + np.cos(np.pi * progress))
    return 0.1*lr + 0.9*lr * cosine

def get_alpha(epoch, config):
    final_epoch = config['final_epoch']
    final_alpha = config['final_alpha']
    if epoch >= final_epoch:
        return final_alpha
    progress = epoch / final_epoch
    return final_alpha * 0.5 * (1 - np.cos(np.pi * progress))

def param_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / (1024 * 1024)  # Convert to millions

def save_img(m, f, name, accelerator, config, epoch):
    mod = m.module if hasattr(m, 'module') else m
    img = f.sample(mod, accelerator.device, use_cond=config['use_cond'])
    save_image(make_grid(img, nrow=10), f'Compare_Reports/images/{name}_epoch_{epoch+1}.png')

def train(model, add_model, flow, add_flow, dataloader, accelerator, config, alpha_config):
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config['learning_rate'], 
                                weight_decay=config['weight_decay'], 
                                betas=config['betas'])
    add_optimizer = torch.optim.Adam(add_model.parameters(),
                                     lr=config['learning_rate'], 
                                    weight_decay=config['weight_decay'], 
                                    betas=config['betas'])
    model, add_model, optimizer, add_optimizer, dataloader = accelerator.prepare(model, add_model, optimizer, add_optimizer, dataloader)
    ema, add_ema = EMA(model, decay=0.99995), EMA(add_model, decay=0.99995)
    
    losses, add_losses = [], []
    alphas, lrs = [], []
    fids, add_fids = [], []
    best_fid, best_add_fid = float('inf'), float('inf')
    
    n_epochs = config['num_epochs']
    for epoch in range(n_epochs):
        start_time = time.time()
        lr = get_lr(epoch, config)
        cur_alpha = get_alpha(epoch, alpha_config)
        alphas.append(cur_alpha)
        
        model.train()
        add_model.train()
        
        epoch_loss, epoch_add_loss = [], []
        for x, c in dataloader:
            loss = flow.loss(model, x, c)
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            ema.update()
            epoch_loss.append(loss.item())
            
            add_loss = add_flow.loss(add_model, x, c, cur_alpha)
            accelerator.backward(add_loss)
            add_optimizer.step()
            add_optimizer.zero_grad()
            add_ema.update()
            epoch_add_loss.append(add_loss.item())
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        for param_group in add_optimizer.param_groups:
            param_group['lr'] = lr
        lrs.append(lr)
        
        losses.append(np.mean(epoch_loss))
        add_losses.append(np.mean(epoch_add_loss))
        
        epoch_time = time.time() - start_time
        
        if accelerator.is_main_process and (epoch + 1) % config['log_epoch'] == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {losses[-1]:.4f},AddLoss: {add_losses[-1]:.4f}, LR: {lrs[-1]:.6f}, Alpha: {cur_alpha:.2f}, Time: {epoch_time:.2f}s")
        
        if (epoch + 1) % config['sample_epoch'] == 0:
            ema.apply_shadow()
            add_ema.apply_shadow()
            model.eval()
            add_model.eval()
            with torch.no_grad():
                save_img(model, flow, "MeanFlow", accelerator, config, epoch)
                save_img(add_model, add_flow, "AdditiveMeanFlow", accelerator, config, epoch)
                
                begin_time = time.time()
                mod = model.module if hasattr(model, 'module') else model
                add_mod = add_model.module if hasattr(add_model, 'module') else add_model
                fake_imgs = flow.sample(mod, accelerator.device, use_cond=config['use_cond'], n=2000)
                add_fake_imgs = add_flow.sample(add_mod, accelerator.device, use_cond=config['use_cond'], n=2000)
                # 加载真实图片
                real_images = get_real_images(
                    DATA_CONFIG['dataset_type'],
                    DATA_CONFIG['image_size'],
                    batch_size=256,
                    num_images=10000
                ).to(accelerator.device)
            fid = calculate_fid(real_images=real_images, fake_images=fake_imgs, device=accelerator.device)
            add_fid = calculate_fid(real_images=real_images, fake_images=add_fake_imgs, device=accelerator.device)
            end_time = time.time() - begin_time
            fids.append(fid)
            add_fids.append(add_fid)
            if accelerator.is_main_process:
                print(f"Epoch {epoch+1}: MeanFlow FID={fid:.2f}, AdditiveMeanFlow FID={add_fid:.2f}, Time: {end_time:.2f}s")
            if fid < best_fid:
                best_fid = fid
                accelerator.save_state(f'Compare_Reports/best_meanflow.pth')
                if accelerator.is_main_process:
                    print(f"New best MeanFlow FID: {best_fid:.2f}, model saved.")
            if add_fid < best_add_fid:
                best_add_fid = add_fid
                accelerator.save_state(f'Compare_Reports/best_additive_meanflow.pth')
                if accelerator.is_main_process:
                    print(f"New best AdditiveMeanFlow FID: {best_add_fid:.2f}, model saved.")
            torch.cuda.empty_cache()
        
        ema.restore()
        add_ema.restore()
        accelerator.wait_for_everyone()
    return model, add_model, losses, add_losses, alphas, lrs, fids, add_fids

def create_models():
    model = MFDiT(**DIT_CONFIG)
    add_model = MFDiT(**DIT_CONFIG)
    flow = MeanFlow(**MEANFLOW_CONFIG)
    add_flow = AdditiveMeanFlow(**MEANFLOW_CONFIG)
    return model, add_model, flow, add_flow

def setup_dirs():
    for d in ['Compare_Reports', 'Compare_Reports/images']:
        os.makedirs(d, exist_ok=True)

def main():
    setup_dirs()
    accelerator = Accelerator(mixed_precision=EXPERIMENT_CONFIG['mixed_precision'])
    dataloader = create_dataloader(DATA_CONFIG)
    model, add_model, flow, add_flow = create_models()
    
    if accelerator.is_main_process:
        print(f"Dataset: {DATA_CONFIG['dataset_type']}, Image Size: {DIT_CONFIG['input_size']}, Alpha: 0->{ALPHA_CONFIG['final_alpha']}, Params: {param_count(model):.2f}M")
    
    model, add_model, losses, add_losses, alphas, lrs, fids, add_fids, best_fid, best_add_fid = train(
        model, add_model, flow, add_flow, dataloader, accelerator, EXPERIMENT_CONFIG, ALPHA_CONFIG
    )
    
    plot_training_curves(
        losses, add_losses, fids, add_fids, alphas, lrs,
        save_path='Compare_Reports/training_curves.png'
    )
    
    best_fid = min(fids) if fids else float('inf')
    best_add_fid = min(add_fids) if add_fids else float('inf')
    best_fid_epoch = fids.index(best_fid) + 1 if fids else -1
    best_add_fid_epoch = add_fids.index(best_add_fid) + 1 if add_fids else -1
    
    write_training_report(
        save_path='Compare_Reports/training_report.md',
        dataset_name=DATA_CONFIG['dataset_type'],
        model_name1='MeanFlow',
        model_name2='AdditiveMeanFlow',
        best_fid1=best_fid,
        best_fid1_epoch=best_fid_epoch,
        best_fid2=best_add_fid,
        best_fid2_epoch=best_add_fid_epoch,
        data_config=DATA_CONFIG,
        exp_config=EXPERIMENT_CONFIG,
        dit_config=DIT_CONFIG,
        meanflow_config=MEANFLOW_CONFIG,
        alpha_config=ALPHA_CONFIG
    )

if __name__ == "__main__":
    main()