import os
import sys
import unittest
import numpy as np
import torch

# Add project root to path
sys.path.append(os.getcwd())

from src.dataset import load_eurasat, naive_random_split, block_split
from src.model import TransferLearningModel
from src.change_detection import compute_cosine_similarity
from src.gradcam import GradCAM

class TestSatellitePipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cls.model = TransferLearningModel(num_classes=10, backbone_name="resnet18").to(cls.device)
        cls.model.eval()
        
    def test_model_dimensions(self):
        # Batch of 4 dummy images (3 channels, 64x64 pixels)
        dummy_input = torch.randn(4, 3, 64, 64).to(self.device)
        with torch.no_grad():
            output = self.model(dummy_input)
        self.assertEqual(output.shape, (4, 10), "Model logit output shape must be (Batch, 10)")

    def test_model_freezing(self):
        # Verify backbone freezing
        self.model.freeze_backbone()
        for name, param in self.model.backbone.named_parameters():
            if "fc" in name:
                self.assertTrue(param.requires_grad, "Classifier head fc layer must remain trainable")
            else:
                self.assertFalse(param.requires_grad, "Backbone parameters must be frozen")
        
        # Verify backbone unfreezing of blocks (conv1 and bn1 remain frozen to preserve low-level features)
        self.model.unfreeze_last_blocks(n_blocks=4)
        for name, param in self.model.backbone.named_parameters():
            if any(layer in name for layer in ["layer1", "layer2", "layer3", "layer4", "fc"]):
                self.assertTrue(param.requires_grad, f"Unfrozen layer {name} must be trainable")

    def test_split_strategy(self):
        # Mock dataset labels representing EuroSAT-scale data (2,700 samples per class, 10 classes)
        mock_labels = np.array([i for i in range(10) for _ in range(2700)])
        splits = (0.7, 0.15, 0.15)
        
        # Naive split
        naive = naive_random_split(len(mock_labels), splits=splits, seed=42)
        self.assertEqual(len(naive['train']) + len(naive['val']) + len(naive['test']), len(mock_labels))
        self.assertTrue(len(naive['train']) > 0)
        self.assertTrue(len(naive['val']) > 0)
        self.assertTrue(len(naive['test']) > 0)
        
        # Block split
        blocked = block_split(mock_labels, splits=splits, block_size=40, seed=42)
        self.assertEqual(len(blocked['train']) + len(blocked['val']) + len(blocked['test']), len(mock_labels))
        self.assertTrue(len(blocked['train']) > 0)
        self.assertTrue(len(blocked['val']) > 0)
        self.assertTrue(len(blocked['test']) > 0)
        
        # Ensure splits are disjoint
        all_indices = set(blocked['train']) | set(blocked['val']) | set(blocked['test'])
        self.assertEqual(len(all_indices), len(mock_labels), "Splits must cover all index space")
        self.assertEqual(len(set(blocked['train']) & set(blocked['val'])), 0, "Train and Val must be disjoint")
        self.assertEqual(len(set(blocked['train']) & set(blocked['test'])), 0, "Train and Test must be disjoint")
        self.assertEqual(len(set(blocked['val']) & set(blocked['test'])), 0, "Val and Test must be disjoint")

    def test_cosine_similarity(self):
        emb1 = np.array([[1.0, 0.0, 0.0]])
        emb2 = np.array([[1.0, 0.0, 0.0]])
        sim = compute_cosine_similarity(emb1, emb2)
        self.assertAlmostEqual(sim[0], 1.0, places=5, msg="Identical vectors must have similarity of 1.0")
        
        emb3 = np.array([[-1.0, 0.0, 0.0]])
        sim_opp = compute_cosine_similarity(emb1, emb3)
        self.assertAlmostEqual(sim_opp[0], -1.0, places=5, msg="Opposite vectors must have similarity of -1.0")

    def test_gradcam_output(self):
        # Initialize GradCAM
        gradcam_target = self.model.backbone.layer4[1].conv2
        cam_model = GradCAM(self.model, gradcam_target)
        
        # Dummy image tensor
        dummy_input = torch.randn(1, 3, 64, 64).to(self.device)
        heatmap, conf = cam_model.generate_heatmap(dummy_input)
        
        self.assertEqual(heatmap.shape, (64, 64), "Grad-CAM heatmap must match input resolution (64, 64)")
        self.assertTrue(0.0 <= conf <= 1.0, "Confidence score must be in range [0, 1]")

    def test_dataset_loading(self):
        data_path = os.path.join("data", "raw")
        if os.path.exists(data_path) and len(os.listdir(data_path)) > 0:
            dataset = load_eurasat(root=data_path, download=False)
            self.assertTrue(len(dataset) > 0, "Dataset must load and contain samples")
            self.assertEqual(len(dataset.classes), 10, "EuroSAT dataset must contain 10 classes")

if __name__ == "__main__":
    unittest.main()
