import sys
sys.path.append('./taming-transformers/')
from omegaconf import OmegaConf
import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image
import numpy as np
from IPython.display import clear_output
import time

from taming.models import cond_transformer, vqgan

import config
from .utils import (
    show_image,
    show_segmentation,
    save_image,
)

class ReplaceGrad(torch.autograd.Function):
    """
    Custom autograd function to replace gradients during the backward pass.

    This is useful in optimization problems where we want to use one tensor for
    the forward pass and another tensor for the backward pass.
    """
    @staticmethod
    def forward(ctx, x_forward, x_backward):
        """
        Forward pass returns x_forward.

        Parameters:
        x_forward (torch.Tensor): Forward tensor.
        x_backward (torch.Tensor): Tensor used for backward gradient.

        Returns:
        torch.Tensor: x_forward tensor.
        """
        # Save the shape of the backward tensor for later use
        ctx.shape = x_backward.shape
        return x_forward

    @staticmethod
    def backward(ctx, grad_in):
        """
        Backward pass returns the sum of gradients for x_backward.

        Parameters:
        grad_in (torch.Tensor): Incoming gradient.

        Returns:
        tuple: None for x_forward, sum of gradients for x_backward.
        """
        # Return None for x_forward and the gradient for x_backward
        return None, grad_in.sum_to_size(ctx.shape)


class ClampWithGrad(torch.autograd.Function):
    """
    Custom autograd function to clamp input values with gradient computation.

    This is useful for clamping values during training while retaining
    the ability to compute gradients for backpropagation.
    """
    @staticmethod
    def forward(ctx, input, min, max):
        """
        Forward pass clamps the input between min and max.

        Parameters:
        input (torch.Tensor): Input tensor.
        min (float): Minimum value.
        max (float): Maximum value.

        Returns:
        torch.Tensor: Clamped tensor.
        """
        # Save min, max, and input tensor for backward pass
        ctx.min = min
        ctx.max = max
        ctx.save_for_backward(input)
        # Clamp the input tensor
        return input.clamp(min, max)

    @staticmethod
    def backward(ctx, grad_in):
        """
        Backward pass computes the gradient for the clamped input.

        Parameters:
        grad_in (torch.Tensor): Incoming gradient.

        Returns:
        tuple: Gradient for input, None for min and max.
        """
        input, = ctx.saved_tensors
        # Compute the gradient only for values within the clamp range
        return grad_in * (grad_in * (input - input.clamp(ctx.min, ctx.max)) >= 0), None, None

    
# Alias for applying the ReplaceGrad function
REPLACE_GRAD = ReplaceGrad.apply

# Alias for applying the ClampWithGrad function
CLAMP_WITH_GRAD = ClampWithGrad.apply
    
    
def load_vqgan_model(config_path, checkpoint_path):
    """
    Load the VQGAN model from configuration and checkpoint files.

    This function initializes the VQGAN model using the provided configuration
    and checkpoint files, setting it up for use in image generation tasks.

    Parameters:
    config_path (str): Path to the configuration file.
    checkpoint_path (str): Path to the model checkpoint file.

    Returns:
    nn.Module: Loaded VQGAN model.
    """
    # Load the model configuration
    config = OmegaConf.load(config_path)

    # Initialize the model based on the configuration target
    model = cond_transformer.Net2NetTransformer(**config.model.params)
    model.eval().requires_grad_(False)
    model.init_from_ckpt(checkpoint_path)
    print('✅ Done!')
    
    return model
    
    
def load_and_process_segmentation(path, plot_segmentation=True, num_categories_expected=183, device='cuda'):
    """
    Load a segmentation image and process it for VQGAN input.

    Args:
    path (str): Path to the segmentation image.
    num_categories_expected (int): Number of categories the model expects.
    device (str): Device to use for tensor operations.

    Returns:
    torch.Tensor: Processed segmentation tensor.
    """
    segmentation = np.array(Image.open(path))
    
    num_categories_in_image = np.max(segmentation) + 1
    print(f"⛳️ Number of categories in segmentation image: {num_categories_in_image}")

    segmentation_one_hot = np.eye(num_categories_in_image)[segmentation]
    segmentation_tensor = torch.tensor(segmentation_one_hot.transpose(2, 0, 1)[None], 
                                       dtype=torch.float32, device=device)

    if num_categories_in_image < num_categories_expected:
        padding = torch.zeros((1, num_categories_expected - num_categories_in_image, 
                               *segmentation_tensor.shape[2:]), device=device)
        segmentation_tensor = torch.cat([segmentation_tensor, padding], dim=1)
    else:
        segmentation_tensor = segmentation_tensor[:, :num_categories_expected, :, :]

    print(f"⛳️ Segmentation tensor shape: {segmentation_tensor.shape}")
    
    if plot_segmentation:
        show_segmentation(segmentation_tensor)
        
    return segmentation_tensor


def generate_iteration(model, c_code, c_indices, z_indices, iteration=0, temperature=1.0, top_k=100, update_every=50):
    """
    Generate an image using VQGAN.
    
    Args:
    model (torch.nn.Module): The VQGAN model.
    c_code (torch.Tensor): Conditional code.
    c_indices (torch.Tensor): Conditional indices.
    z_indices (torch.Tensor): Initial z indices.
    temperature (float): Temperature for softmax.
    top_k (int): Top-k sampling parameter.
    update_every (int): Frequency of image updates.
    
    Returns:
    torch.Tensor: The generated image.
    """
    z_code_shape = c_code.shape
    idx = z_indices.reshape(z_code_shape[0], z_code_shape[2], z_code_shape[3])
    cidx = c_indices.reshape(c_code.shape[0], c_code.shape[2], c_code.shape[3])
    
    start_t = time.time()
    
    for i in range(z_code_shape[2]):
        if i <= 8:
            local_i = i
        elif z_code_shape[2] - i < 8:
            local_i = 16 - (z_code_shape[2] - i)
        else:
            local_i = 8
        
        for j in range(z_code_shape[3]):
            if j <= 8:
                local_j = j
            elif z_code_shape[3] - j < 8:
                local_j = 16 - (z_code_shape[3] - j)
            else:
                local_j = 8
            
            i_start = i - local_i
            i_end = i_start + 16
            j_start = j - local_j
            j_end = j_start + 16
            
            patch = idx[:, i_start:i_end, j_start:j_end]
            patch = patch.reshape(patch.shape[0], -1)
            cpatch = cidx[:, i_start:i_end, j_start:j_end]
            cpatch = cpatch.reshape(cpatch.shape[0], -1)
            patch = torch.cat((cpatch, patch), dim=1)
            
            logits, _ = model.transformer(patch[:, :-1])
            logits = logits[:, -256:, :]
            logits = logits.reshape(z_code_shape[0], 16, 16, -1)
            logits = logits[:, local_i, local_j, :]
            logits = logits / temperature
            
            if top_k is not None:
                logits = model.top_k_logits(logits, top_k)
            
            probs = F.softmax(logits, dim=-1)
            idx[:, i, j] = torch.multinomial(probs, num_samples=1)
            
            step = i * z_code_shape[3] + j
            if step % update_every == 0 or step == z_code_shape[2] * z_code_shape[3] - 1:
                x_sample = model.decode_to_img(idx, z_code_shape)
                save_image(x_sample, step, iteration, sampling_folder='sampling')
                clear_output(wait=True)
                print(f"🕰️ Time: {time.time() - start_t:.2f} seconds")
                print(f"⛳️ Step: ({i},{j}) | Local: ({local_i},{local_j}) | Crop: ({i_start}:{i_end},{j_start}:{j_end})")
                show_image(x_sample)
    
    return model.decode_to_img(idx, z_code_shape)


def generate_image(model, config, num_iterations=3, conditional=True):
    """
    Main function to run the image generation process.
    Args:
    model (torch.nn.Module): The VQGAN model.
    config: Configuration object containing parameters.
    conditional (bool): Whether to use conditional generation.
    num_iterations (int): Number of iterations to run.
    """
    if conditional:
        segmentation_tensor = load_and_process_segmentation(
            config.INITIAL_IMAGE,
            plot_segmentation=True,
            device=model.device,
        )
        c_code, c_indices = model.encode_to_c(segmentation_tensor)
        
    else:
        random_input = create_random_input(
            (config.HEIGHT, config.WIDTH), 
            183,
            model.device,
        )
        c_code, c_indices = model.encode_to_c(random_input)
        
    print("c_code", c_code.shape, c_code.dtype)
    print("c_indices", c_indices.shape, c_indices.dtype)
    
    z_indices = torch.randint(256, c_indices.shape, device=model.device)
    initial_image = model.decode_to_img(z_indices, c_code.shape)
    
    print("Initial random image:")
    show_image(initial_image)
    
    for iteration in range(num_iterations):
        print(f"⛳️ Starting iteration {iteration + 1}/{num_iterations}")
        final_image = generate_iteration(
            model, 
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
