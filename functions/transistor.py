import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
from functools import partial
from torch.utils.tensorboard import SummaryWriter

from functions.t5 import encode_text
from functions.utils import (
    show_segmentation, 
    show_image, 
    clear_folder, 
    create_gif,
)
from functions.vqgan import (
    load_process_encode_rgb_image, 
    generate_iteration, 
    load_and_process_segmentation,
)
import config

def generate_image_from_text(user_query, transistor_model, vqgan_model, encode_text_fn, device, n_iterations=1, sampling_folder='sampling'):
    
    clear_folder('sampling')
    
    # Encode text
    text_latent, _ = encode_text_fn([user_query])
    text_latent = text_latent.mean(dim=1).to(device)  # Average over token dimension

    # Pass through Transistor
    with torch.no_grad():
        image_latent = transistor_model(text_latent)

    # Reshape
    image_latent = image_latent.view(1, 256, 16, 16)

    # Quantize (this step depends on VQGAN's specific implementation)
    c_code, _, [_, _, c_indices] = vqgan_model.first_stage_model.quantize(image_latent)
    
    print("c_code", c_code.shape, c_code.dtype)
    print("c_indices", c_indices.shape, c_indices.dtype)

    z_indices = torch.randint(256, c_indices.shape, device=vqgan_model.device)
    initial_image = vqgan_model.decode_to_img(z_indices, c_code.shape)

    print("🖼️ Initial random image:")
    show_image(initial_image)

    for iteration in range(n_iterations):
        print(f"⛳️ Starting iteration {iteration + 1}/{n_iterations}")
        final_image = generate_iteration(
            vqgan_model, 
            c_code, 
            c_indices, 
            z_indices, 
            temperature=config.TEMPERATURE, 
            top_k=config.TOP_K, 
            update_every=config.UPDATE_EVERY,
            iteration=iteration,
            sampling_folder=sampling_folder,
        )

    print("✅ All iterations completed.")
    
    return final_image


def generate_image_from_text_mask(user_query, mask_path, transistor_model, vqgan_model, encode_text_fn, device, n_iterations=1):
    
    clear_folder('sampling')
    
    segmentation_tensor = load_and_process_segmentation(
        mask_path,
        plot_segmentation=True,
        device=vqgan_model.device,
        num_categories_expected=182,
        target_size=(256, 256),
    )        
    c_code_mask, c_indices_mask = vqgan_model.encode_to_c(segmentation_tensor)
    
    # Encode text
    text_latent, _ = encode_text_fn([user_query])
    text_latent = text_latent.mean(dim=1).to(device)  # Average over token dimension

    # Pass through Transistor
    with torch.no_grad():
        image_latent = transistor_model(text_latent)

    # Reshape
    image_latent = image_latent.view(1, 256, 16, 16)

    # Quantize (this step depends on VQGAN's specific implementation)
    c_code_text, _, [_, _, c_indices_text] = vqgan_model.first_stage_model.quantize(image_latent)

    z_indices = torch.randint(256, c_indices_text.shape, device=vqgan_model.device)
    initial_image = vqgan_model.decode_to_img(z_indices, c_code_text.shape)

    print("🖼️ Initial random image:")
    show_image(initial_image)

    for iteration in range(n_iterations):
        print(f"⛳️ Starting iteration {iteration + 1}/{n_iterations}")
        
        if iteration % 2 == 0:
            c_code, c_indices = c_code_text, c_indices_text
        
        else:
            c_code, c_indices = c_code_mask, c_indices_mask
        
        final_image = generate_iteration(
            vqgan_model, 
            c_code, 
            c_indices, 
            z_indices, 
            temperature=config.TEMPERATURE, 
            top_k=config.TOP_K, 
            update_every=config.UPDATE_EVERY,
            iteration=iteration,
        )

    print("✅ All iterations completed.")
    
    return final_image


def generate_image_from_text_mask_weighted(user_query, mask_path, transistor_model, vqgan_model, encode_text_fn, device, n_iterations=5, initial_text_weight=0.3, final_text_weight=0.7):
    
    if mask_path:
        segmentation_tensor = load_and_process_segmentation(
            mask_path,
            plot_segmentation=True,
            device=vqgan_model.device,
            num_categories_expected=182,
            target_size=(256, 256),
        )        
        c_code_mask, c_indices_mask = vqgan_model.encode_to_c(segmentation_tensor)
    
    # Encode text
    text_latent, _ = encode_text_fn([user_query])
    text_latent = text_latent.mean(dim=1).to(device)  # Average over token dimension

    # Pass through Transistor
    with torch.no_grad():
        image_latent = transistor_model(text_latent)

    # Reshape
    image_latent = image_latent.view(1, 256, 16, 16)

    # Quantize (this step depends on VQGAN's specific implementation)
    c_code_text, _, [_, _, c_indices_text] = vqgan_model.first_stage_model.quantize(image_latent)
    
    print("c_code_text", c_code_text.shape, c_code_text.dtype)
    print("c_indices_text", c_indices_text.shape, c_indices_text.dtype)
    
    print("c_code_mask", c_code_mask.shape, c_code_mask.dtype)
    print("c_indices_mask", c_indices_mask.shape, c_indices_mask.dtype)
    
    c_code_combined = (c_code_mask + c_code_text) / 2
    c_indices_combined = (c_indices_mask + c_indices_text) / 2

    z_indices = torch.randint(256, c_indices_text.shape, device=vqgan_model.device)
    initial_image = vqgan_model.decode_to_img(z_indices, c_code_text.shape)

    print("Initial random image:")
    show_image(initial_image)

    for iteration in range(n_iterations):
        print(f"⛳️ Starting iteration {iteration + 1}/{n_iterations}")
        
        # Calculate a dynamic weight that gradually increases the text influence
        if n_iterations > 1:
            text_weight = initial_text_weight + (final_text_weight - initial_text_weight) * (iteration / (n_iterations - 1))
        else:
            text_weight = final_text_weight
        
        # Combine latents with dynamic weighting
        combined_c_code = (1 - text_weight) * c_code_mask + text_weight * c_code_text
        
        # For c_indices, we'll use the mask indices as a base and gradually introduce text indices
        combined_c_indices = torch.where(
            torch.rand_like(c_indices_mask.float()) < text_weight,
            c_indices_text,
            c_indices_mask
        )

        final_image = generate_iteration(
            vqgan_model, 
            combined_c_code, 
            combined_c_indices, 
            z_indices, 
            temperature=config.TEMPERATURE * (1 + iteration / max(n_iterations - 1, 1)),  # Avoid division by zero
            top_k=max(config.TOP_K - iteration, 1),  # Gradually decrease top_k
            update_every=config.UPDATE_EVERY,
            iteration=iteration,
        )
        
    print("✅ All iterations completed.")
    return final_image