import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from PIL import Image
from pycocotools.coco import COCO
from tqdm import tqdm
    
    
def plot_original_and_annotated(coco, img, img_dir):
    """
    Plot original image and annotated image side by side.
    
    Args:
    coco: COCO object
    img: Image metadata dictionary
    img_dir: Directory containing the images
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Original image
    image = np.array(Image.open(os.path.join(img_dir, img['file_name'])))
    ax1.imshow(image)
    ax1.set_title('Original Image', fontsize=20)
    ax1.axis('off')
    
    # Annotated image
    ax2.imshow(image)
    cat_ids = coco.getCatIds()
    anns_ids = coco.getAnnIds(imgIds=img['id'], catIds=cat_ids, iscrowd=None)
    anns = coco.loadAnns(anns_ids)
    
    # Draw annotations manually
    polygons = []
    color = []
    for ann in anns:
        c = (np.random.random((1, 3))*0.6+0.4).tolist()[0]
        if 'segmentation' in ann:
            if type(ann['segmentation']) == list:
                # polygon
                for seg in ann['segmentation']:
                    poly = np.array(seg).reshape((int(len(seg)/2), 2))
                    polygons.append(Polygon(poly))
                    color.append(c)
    
    p = PatchCollection(polygons, facecolor=color, linewidths=0, alpha=0.4)
    ax2.add_collection(p)
    p = PatchCollection(polygons, facecolor='none', edgecolors=color, linewidths=2)
    ax2.add_collection(p)
    
    ax2.set_title('Annotated Image', fontsize=20)
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()

    
def plot_original_and_segmentation(coco, img, img_dir):
    """
    Plot original image and segmentation mask side by side.
    
    Args:
    coco: COCO object
    img: Image metadata dictionary
    img_dir: Directory containing the images
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Original image
    image = np.array(Image.open(os.path.join(img_dir, img['file_name'])))
    ax1.imshow(image)
    ax1.set_title('Original Image', fontsize=20)
    ax1.axis('off')
    
    # Segmentation mask
    cat_ids = coco.getCatIds()
    anns_ids = coco.getAnnIds(imgIds=img['id'], catIds=cat_ids, iscrowd=None)
    anns = coco.loadAnns(anns_ids)
    anns_img = np.zeros((img['height'], img['width']))
    for ann in anns:
        anns_img = np.maximum(anns_img, coco.annToMask(ann) * ann['category_id'])
    
    ax2.imshow(anns_img)
    ax2.set_title('Segmentation Mask', fontsize=20)
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()
    

def create_and_save_segmentation_masks(coco, img_dir, seg_dir, display_samples=False):
    """
    Iterate through each image, create a segmentation mask and save it as a colored image.
    
    Args:
    coco: COCO object
    img_dir: Directory containing the images
    seg_dir: Directory to save segmentation masks
    display_samples: If True, display some sample masks during processing
    """
    # Create segmentation directory if it doesn't exist
    os.makedirs(seg_dir, exist_ok=True)
    
    # Get total number of images
    total_images = len(coco.imgs)
    
    # Create a colormap
    cmap = plt.get_cmap('viridis')
    
    # Create tqdm progress bar
    with tqdm(total=total_images, desc="🧬 Creating segmentation masks") as pbar:
        for i, img_id in enumerate(coco.imgs):
            img = coco.imgs[img_id]
            
            # Create segmentation mask using the specified method
            cat_ids = coco.getCatIds()
            anns_ids = coco.getAnnIds(imgIds=img['id'], catIds=cat_ids, iscrowd=None)
            anns = coco.loadAnns(anns_ids)
            anns_img = np.zeros((img['height'], img['width']))
            for ann in anns:
                anns_img = np.maximum(anns_img, coco.annToMask(ann) * ann['category_id'])
            
            # Convert the mask to a colored image
            colored_mask = cmap(anns_img / anns_img.max())[:, :, :3]
            colored_mask = (colored_mask * 255).astype(np.uint8)
            
            # Display some sample masks
            if display_samples and i % 100 == 0:
                plt.figure(figsize=(10, 10))
                plt.imshow(colored_mask)
                plt.title(f"Segmentation mask for {img['file_name']}")
                plt.axis('off')
                plt.show()
            
            # Save colored segmentation mask
            seg_filename = os.path.splitext(img['file_name'])[0] + '_seg.png'
            seg_path = os.path.join(seg_dir, seg_filename)
            Image.fromarray(colored_mask).save(seg_path)
            
            # Update progress bar
            pbar.update(1)
            pbar.set_postfix({'⛳️ Current image': img['file_name']})
    
    print(f"✅ Completed! All colored segmentation masks saved in {seg_dir}")
