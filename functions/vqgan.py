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
import matplotlib.pyplot as plt

from taming.models import cond_transformer, vqgan
import config
from .utils import (
    show_image,
    show_segmentation,
    save_image,
)
    
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
    
     
def load_and_process_segmentation(path, plot_segmentation=True, num_categories_expected=183, device='cuda', target_size=None):
    """
    Load a segmentation image, process it for VQGAN input, and resize to target dimensions.
    Args:
    path (str): Path to the segmentation image.
    plot_segmentation (bool): Whether to plot the segmentation.
    num_categories_expected (int): Number of categories the model expects.
    device (str): Device to use for tensor operations.
    target_size (tuple): Target size for the segmentation tensor (height, width).
    Returns:
    torch.Tensor: Processed and resized segmentation tensor.
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
    
    if target_size:
        # Resize the segmentation tensor to the target size
        segmentation_tensor = F.interpolate(segmentation_tensor, size=target_size, mode='nearest')

        print(f"⛳️ Resized segmentation tensor shape: {segmentation_tensor.shape}")
    
    if plot_segmentation:
        show_segmentation(segmentation_tensor)
        
    return segmentation_tensor


def generate_iteration(
    model, c_code, c_indices, z_indices, iteration=0, temperature=1.0, top_k=100, update_every=50, sampling_folder='sampling'
):
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
                save_image(x_sample, step, iteration, sampling_folder=sampling_folder)
                clear_output(wait=True)
                print(f"🕰️ Time: {time.time() - start_t:.2f} seconds")
                print(f"⛳️ Step: ({i},{j}) | Local: ({local_i},{local_j}) | Crop: ({i_start}:{i_end},{j_start}:{j_end})")
                show_image(x_sample)
    
    return model.decode_to_img(idx, z_code_shape)


def generate_image(model, config, num_iterations=3, conditional=True, num_categories_expected=183, sampling_folder='sampling'):
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
            num_categories_expected=num_categories_expected,
        )
        c_code, c_indices = model.encode_to_c(segmentation_tensor)
        
    else:
        random_input = create_random_input(
            (config.HEIGHT, config.WIDTH), 
            num_categories_expected,
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
            sampling_folder=sampling_folder,
        )

    print("✅ All iterations completed.")
    return final_image

def preprocess_vqgan(x):
    x = 2. * x - 1.
    return x

def custom_to_pil(x):
    x = x.detach().cpu()
    x = torch.clamp(x, -1., 1.)
    x = (x + 1.) / 2.
    x = x.permute(1, 2, 0).numpy()
    x = (255 * x).astype(np.uint8)
    x = Image.fromarray(x)
    if not x.mode == "RGB":
        x = x.convert("RGB")
    return x
    
    
def load_process_encode_rgb_image(path, model, return_reconstruct=False, target_size=256, device='cuda'):
    """
    Load an RGB image, process it, and encode it using the VQGAN encoder.

    Args:
    path (str): Path to the RGB image.
    model (torch.nn.Module): The VQGAN model.
    target_size (int): Target size for the image (both height and width).
    device (str): Device to use for tensor operations.

    Returns:
    tuple: (z, c_indices, x_rec)
        z (torch.Tensor): Encoded latent representation.
        c_indices (torch.Tensor): Indices from the quantized representation.
        x_rec (torch.Tensor): Reconstructed image tensor.
    """


    # Load and preprocess the image
    img = Image.open(path).convert('RGB')
    s = min(img.size)
    if s < target_size:
        raise ValueError(f'min dim for image {s} < {target_size}')
    r = target_size / s
    s = (round(r * img.size[1]), round(r * img.size[0]))
    img = TF.resize(img, s, interpolation=Image.LANCZOS)
    img = TF.center_crop(img, output_size=2 * [target_size])
    x = transforms.ToTensor()(img).unsqueeze(0).to(device)

    # Preprocess for VQGAN
    x = preprocess_vqgan(x)

    # Encode the image
    z, _, [_, _, c_indices] = model.first_stage_model.encode(x)
    
    if return_reconstruct:

        # Reconstruct the image (optional, for visualization)
        x_reconstructed = model.first_stage_model.decode(z)

        return z, c_indices, x_reconstructed
    
    return z, c_indices


def show_original_and_reconstruction(original_path, x_rec):
    """
    Display the original image and its reconstruction side by side.

    Args:
    original_path (str): Path to the original image.
    x_rec (torch.Tensor): Reconstructed image tensor.
    """
    # Load original image
    original_img = Image.open(original_path).convert('RGB')

    # Convert reconstructed tensor to PIL image
    rec_img = custom_to_pil(x_rec[0])

    # Display images side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    ax1.imshow(original_img)
    ax1.set_title('Original Image')
    ax1.axis('off')
    ax2.imshow(rec_img)
    ax2.set_title('Reconstructed Image')
    ax2.axis('off')
    plt.show()


