from torch import nn
from torch.utils.data import DataLoader, random_split
import torch.optim as optim

from FAS.core.data import full_dataset
from FAS.core.DiVT import model

# Split Data
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Optimizer & Loss
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
criterion_binary = nn.BCELoss()  # For Real vs Fake
criterion_class = nn.CrossEntropyLoss()  # For Attack Type & Domain


def train_one_epoch(epoch_index):
    model.train()
    running_loss = 0.0

    for i, data in enumerate(train_loader):
        inputs, labels_type, labels_binary = data
        inputs = inputs.to(device)
        labels_type = labels_type.to(device)  # e.g., 0=Real, 1=Print, 2=Replay
        labels_binary = labels_binary.to(device).unsqueeze(1)  # 0=Real, 1=Fake

        optimizer.zero_grad()

        # Forward Pass
        # alpha controls how much we "unlearn" domain features. Increases over time.
        alpha = max(0, min(1, epoch_index / 20.0))

        live_out, attack_out, domain_out = model(inputs, alpha)

        # Calculate Losses
        loss_live = criterion_binary(live_out, labels_binary)
        loss_attack = criterion_class(attack_out, labels_type)

        # Domain Loss (We want to minimize this for the discriminator,
        # but the GRL reverses it for the backbone)
        loss_domain = criterion_class(domain_out, labels_type)

        # Total Loss
        # We weigh liveness highest as it is the primary safety goal
        total_loss = loss_live + (0.5 * loss_attack) + (0.1 * loss_domain)

        total_loss.backward()
        optimizer.step()

        running_loss += total_loss.item()

        if i % 10 == 0:
            print(f"Epoch {epoch_index}, Batch {i}, Loss: {total_loss.item():.4f}")


# Run Training
for epoch in range(5):
    print(f"--- Starting Epoch {epoch} ---")
    train_one_epoch(epoch)