import torch
import cv2
import os
import pandas as pd
from skimage import io
from torchvision import transforms
from torchvision.models.vision_transformer import vit_b_16 as ViT, ViT_B_16_Weights
import argparse

# ==========================
# 1️⃣ Set Device (GPU or CPU)
# ==========================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# ==========================
# 2️⃣ Load Class Labels
# ==========================
def load_class_labels(csv_path):
    """Loads class index to label mapping from sports.csv."""
    df = pd.read_csv(csv_path)
    class_dict = {row["class id"]: row["labels"] for _, row in df.iterrows()}
    return class_dict

# Modify this path if needed
class_file = "dataset/sports.csv"
class_labels = load_class_labels(class_file)

# ==========================
# 3️⃣ Load Pretrained ViT Model
# ==========================
def load_model(checkpoint_path):
    """Loads the Vision Transformer model with trained weights."""
    model = ViT(weights=ViT_B_16_Weights.DEFAULT)
    model.heads = torch.nn.Sequential(torch.nn.Linear(in_features=768, out_features=100, bias=True))
    model.to(device)

    # Load checkpoint safely
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    print("✅ Model Loaded Successfully!")
    return model

# ==========================
# 4️⃣ Define Data Transformations
# ==========================
val_transform = transforms.Compose([
    transforms.ToPILImage(mode='RGB'),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

# ==========================
# 5️⃣ Predict Function
# ==========================
def predict_image(model, image_path):
    """Load an image, apply transformations, and predict its class."""
    image = io.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) if len(image.shape) == 2 else image
    image = val_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        _, predicted_class = torch.max(output, 1)

    return predicted_class.item()

# ==========================
# 6️⃣ Extract Actual Class from Path
# ==========================
def extract_actual_class(image_path):
    """Extracts the actual class label from the image path."""
    # Assuming the path follows: dataset/test/class_name/image.jpg
    parts = image_path.split('/')
    if len(parts) >= 3:
        actual_label = parts[-2]  # The folder name before the image file
        return actual_label
    return "Unknown"

# ==========================
# 7️⃣ Main Execution for Inference
# ==========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on images using a trained ViT model.")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model checkpoint (.pth)")
    parser.add_argument("--images", type=str, nargs="+", required=True, help="List of image file paths for prediction")
    
    args = parser.parse_args()

    # Load model from checkpoint
    model = load_model(args.model)

    # Open log file to store inference results
    result_file = "inference_results.txt"

    with open(result_file, "w") as f:
        # Run inference on provided images
        for img_path in args.images:
            if not os.path.exists(img_path):
                print(f"❌ Error: Image {img_path} not found!")
                f.write(f"❌ Error: Image {img_path} not found!\n")
                continue
            
            predicted_class = predict_image(model, img_path)

            # Get actual class from folder name
            actual_label = extract_actual_class(img_path)

            # Get predicted class label from the mapping
            predicted_label = class_labels.get(predicted_class, "Unknown")

            result = (
                f"✅ Image: {img_path} | "
                f"Actual Class: {actual_label} | "
                f"Predicted Class: {predicted_class} ({predicted_label})"
            )

            print(result)
            f.write(result + "\n")

    print(f"✅ Inference Completed! Results saved in {result_file}")
