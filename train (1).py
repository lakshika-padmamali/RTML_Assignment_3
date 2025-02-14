import pandas as pd
import time
from tqdm import tqdm
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.models.vision_transformer import vit_b_16 as ViT, ViT_B_16_Weights
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset import SportDataset
from logger import Logger
import torch.optim as optim

import torch
import torch.nn as nn
from torchvision.models import vit_b_16 as ViT, ViT_B_16_Weights

# Load the Vision Transformer model
model = ViT(weights=ViT_B_16_Weights.DEFAULT)
model.heads = nn.Sequential(nn.Linear(in_features=768, out_features=100, bias=True))

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)


# Load pretrained weights
checkpoint = torch.load("Ep.7.pth", map_location=device, weights_only=True)
model.load_state_dict(checkpoint)


print("Checkpoint Loaded Successfully!")



from torchvision import transforms

# Augmentation for training
train_transform = transforms.Compose([
    transforms.ToPILImage(mode='RGB'),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
])

# Validation transformation (No Augmentation)
val_transform = transforms.Compose([
    transforms.ToPILImage(mode='RGB'),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])


from dataset import SportDataset
from torch.utils.data import DataLoader

csv_file = "/home/jupyter-st124872/RTML/A3/dataset/sports.csv"
class_file = "/home/jupyter-st124872/RTML/A3/dataset/sports.csv"
root_dir = "/home/jupyter-st124872/RTML/A3/dataset/"

train_ds = SportDataset(csv_file=csv_file, class_file=class_file, root_dir=root_dir, train=True, transform=train_transform)
val_ds = SportDataset(csv_file=csv_file, class_file=class_file, root_dir=root_dir, train=False, transform=val_transform)
batch_size = 8
train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)

#lr = 1e-8
#epoch_number = 0 # describe the starting epoch if you are continuing training
#EPOCHS = 5 # number of epochs to train
# Training Config
start_epoch = 8  # Start from epoch 8 if continuing training
EPOCHS = 30  # Train beyond 30 epochs to beat 94% accuracy
model_name = 'vit_b16'
dataset_name = 'sport_dataset'

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=3e-5, weight_decay=1e-4)
logger = Logger(model_name, dataset_name)

# Learning Rate Scheduler
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

#best_vloss = 100000.
best_vloss = float('inf')
# timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
timestamp = time.time()

# Create a log file to store training results
log_file = "training_log.txt"

with open(log_file, "w") as log:
    for epoch in range(start_epoch, EPOCHS + 1):
        print('EPOCH {}:'.format(epoch))
        log.write(f"\nEPOCH {epoch}/{EPOCHS}\n")
        since = time.time()

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        running_loss = 0.
        last_loss = 0.
        running_acc = 0.
        train_loop = tqdm(train_loader)
        
        # Here, we use enumerate(training_loader) instead of
        # iter(training_loader) so that we can track the batch
        # index and do some intra-epoch reporting
        for i, data in enumerate(train_loop):
            # Every data instance is an input + label pair
            inputs, labels = data['image'].to(device), data['labels'].long().to(device)

            # Zero your gradients for every batch!
            optimizer.zero_grad()

            # Make predictions for this batch
            outputs = model(inputs)

            # Compute the loss and its gradients
            loss = loss_fn(outputs, labels)
            loss.backward()

            # print(labels.shape, outputs.shape)
            _, prediction = torch.max(outputs, dim=1)
            corrects = (labels == (prediction)).sum() / len(labels)
            running_acc += corrects

            # Adjust learning weights
            optimizer.step()

            # Gather data and report
            running_loss += loss.item()

            train_loop.set_postfix(loss=loss.item())

        avg_train_acc = running_acc / len(train_loader)
        avg_train_loss = running_loss / len(train_loader)

        print('Epoch {} loss: {}'.format(epoch, avg_train_loss))
        log.write(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}\n")

        # Learning Rate Scheduler Step
        scheduler.step()
        print(f"Updated Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        log.write(f"Updated Learning Rate: {scheduler.get_last_lr()[0]:.6f}\n")

        # We don't need gradients on to do reporting
        model.train(False)

        vloop = tqdm(val_loader)
        running_vloss = 0.0
        running_vacc = 0.0
        for i, data in enumerate(vloop):
            inputs, labels = data['image'].to(device), data['labels'].long().to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            running_vloss += loss.item()

            _, prediction = torch.max(outputs, dim=1)
            corrects = (prediction == labels).sum() / len(labels)
            running_vacc += corrects

            vloop.set_postfix(loss=loss.item())

        avg_vloss = running_vloss / len(val_loader)
        avg_vacc = running_vacc / len(val_loader)

        print('LOSS train {} valid {}'.format(avg_train_loss, avg_vloss))
        print('Accuracy train {} valid {}'.format(avg_train_acc, avg_vacc))
        
        log.write(f"Valid Loss: {avg_vloss:.4f}, Valid Acc: {avg_vacc:.4f}\n")

        # Log the running loss averaged per batch
        # for both training and validation
        logger.loss_log(train_loss=avg_train_loss,
                        val_loss=avg_vloss, nth_epoch=epoch)

        logger.acc_log(train_acc=avg_train_acc,
                       val_acc=avg_vacc, nth_epoch=epoch)

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = 'model_{}_{}'.format(timestamp, epoch)
            logger.save_models(model=model, nth_epoch=epoch)
            print("✅ Best Model Saved!")
            log.write("✅ Best Model Saved!\n")

        ep_duration = time.time() - since
        print("Epoch time taken: {:.0f}m {:.0f}s".format(ep_duration // 60, ep_duration % 60))
        log.write("Epoch time taken: {:.0f}m {:.0f}s\n".format(ep_duration // 60, ep_duration % 60))

total_time = time.time() - timestamp
print("Total time taken: {:.0f}m {:.0f}s".format(total_time // 60, total_time % 60))

# Log total time taken
with open(log_file, "a") as log:
    log.write("Total time taken: {:.0f}m {:.0f}s\n".format(total_time // 60, total_time % 60))


# Plot Training & Validation Loss/Accuracy
plt.figure(figsize=(12, 5))

# Plot Loss
plt.subplot(1, 2, 1)
plt.plot(range(start_epoch, EPOCHS + 1), train_losses, label='Train Loss', marker='o')
plt.plot(range(start_epoch, EPOCHS + 1), val_losses, label='Validation Loss', marker='o')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.legend()
plt.grid()

# Plot Accuracy
plt.subplot(1, 2, 2)
plt.plot(range(start_epoch, EPOCHS + 1), train_accuracies, label='Train Accuracy', marker='o')
plt.plot(range(start_epoch, EPOCHS + 1), val_accuracies, label='Validation Accuracy', marker='o')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training vs Validation Accuracy')
plt.legend()
plt.grid()

plt.savefig("training_plots.png")
plt.show()

print("🎯 Fine-tuning Completed!")


# GPU 3397MiB
