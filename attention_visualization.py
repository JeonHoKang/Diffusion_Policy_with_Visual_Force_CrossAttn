import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
import numpy as np

def visualize_attention(image, attention_map):
    # Step 1: Aggregate feature map to a single attention map (B, H, W)
    attention_map = attention_map.mean(dim=1)  # (B, H, W)

    # Step 2: Resize attention map to match original image size
    attention_map = F.interpolate(attention_map.unsqueeze(1), size=(image.shape[2], image.shape[3]), mode='bilinear', align_corners=False)
    attention_map = attention_map.squeeze().cpu().detach().numpy()  # Convert to numpy

    # Step 3: Normalize the attention map
    attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())

    # Step 4: Overlay the heatmap on the image
    heatmap = cv2.applyColorMap(np.uint8(255 * attention_map), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    overlayed_image = heatmap * 0.4 + np.float32(image) / 255  # Adjust transparency

    # Step 5: Show the image
    plt.imshow(np.uint8(255 * overlayed_image))
    plt.axis('off')
    plt.show()

# Example usage
image = ...  # Your original image, should be in shape [B, 3, H, W]
cross_attended_features = ...  # Cross-attended feature map in shape [B, C, H, W]

visualize_attention(image[0], cross_attended_features[0])
