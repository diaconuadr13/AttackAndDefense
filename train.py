# train.py
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import SubsetSC, collate_fn
from models.simple_cnn import SimpleAudioCNN

def train(model, loader, optimizer, criterion, device, epoch=None):
    model.train()
    total_loss = 0
    correct = 0
    
    # Create a progress bar
    # leave=False keeps the terminal clean by clearing the bar after each epoch
    desc = f"Epoch {epoch} Training" if epoch else "Training"
    pbar = tqdm(loader, desc=desc, unit="batch", leave=False)
    
    for data, target in pbar:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        
        # Update the progress bar description with the current loss
        pbar.set_postfix(loss=f"{loss.item():.4f}")
        
    return total_loss / len(loader), correct / len(loader.dataset)

def test(model, loader, criterion, device):
    model.eval()
    correct = 0
    
    # Optional: Add a simple progress bar for testing too
    with torch.no_grad():
        for data, target in tqdm(loader, desc="Testing", leave=False):
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            
    return correct / len(loader.dataset)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Data
    train_set = SubsetSC("training")
    test_set = SubsetSC("testing")
    
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, collate_fn=collate_fn, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False, collate_fn=collate_fn, num_workers=0)

    # Init Model
    model = SimpleAudioCNN(n_classes=35).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Training Loop
    for epoch in range(1, 11):
        # We pass the epoch number to the train function for the progress bar label
        loss, acc = train(model, train_loader, optimizer, criterion, device, epoch)
        test_acc = test(model, test_loader, criterion, device)
        print(f"Epoch {epoch}: Loss {loss:.4f}, Train Acc {acc:.4f}, Test Acc {test_acc:.4f}")

    # Save the baseline model
    torch.save(model.state_dict(), "models/baseline_model.pth")
    print("Baseline model saved.")