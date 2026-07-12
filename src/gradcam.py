import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        # Register hooks
        self.forward_hook = self.target_layer.register_forward_hook(self.save_activation)
        self.backward_hook = self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def remove_hooks(self):
        self.forward_hook.remove()
        self.backward_hook.remove()

    def generate_heatmap(self, input_tensor: torch.Tensor, class_idx: int = None):
        """
        Generates a 2D GradCAM heatmap of the input tensor for the given class index.
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        # Zero gradients
        self.model.zero_grad()

        # Backward pass
        loss = output[0, class_idx]
        loss.backward()

        # Get hooked gradients and activations
        gradients = self.gradients.cpu().data.numpy()[0]  # Shape: (C, H, W)
        activations = self.activations.cpu().data.numpy()[0]  # Shape: (C, H, W)

        # Global average pool the gradients
        weights = np.mean(gradients, axis=(1, 2))  # Shape: (C,)

        # Compute the weighted sum of activation maps
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        # Apply ReLU to keep only positive contributions
        cam = np.maximum(cam, 0)

        # Normalize the heatmap to [0, 1]
        if cam.max() > 0:
            cam = cam / cam.max()

        # Resize the heatmap to match input tensor size
        input_h, input_w = input_tensor.shape[2], input_tensor.shape[3]
        cam = cv2.resize(cam, (input_w, input_h))

        return cam, output.softmax(dim=1)[0, class_idx].item()

def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """
    Overlays the 2D GradCAM heatmap onto the original image.
    - image: np.ndarray (H, W, 3) in RGB, scaled [0, 1] or [0, 255]
    - heatmap: np.ndarray (H, W) in range [0, 1]
    """
    # Scale image to [0, 255] if it is [0, 1]
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8)

    # Convert heatmap to [0, 255]
    heatmap_colored = (heatmap * 255).astype(np.uint8)
    
    # Apply colormap
    heatmap_colored = cv2.applyColorMap(heatmap_colored, colormap)
    
    # Convert BGR (OpenCV default) to RGB
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Blend original image and heatmap
    overlayed = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
    return overlayed
