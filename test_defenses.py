import torch
from torch.utils.data import DataLoader
from utils import SubsetSC, collate_fn, get_labels
from models.simple_cnn import SimpleAudioCNN
from attacks.deepfool import deepfool
from defenses.squeezing import FeatureSqueezing

def test_defenses():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device}")
    
    # Setup
    test_set = SubsetSC("testing")
    loader = DataLoader(test_set, batch_size=1, shuffle=True, collate_fn=collate_fn)
    labels = get_labels(test_set)
    
    # Models
    baseline = SimpleAudioCNN(n_classes=35).to(device)
    baseline.load_state_dict(torch.load("models/baseline_model.pth"))
    baseline.eval()
    
    robust = SimpleAudioCNN(n_classes=35).to(device)
    # Only load if you have run train_adv.py
    try:
        robust.load_state_dict(torch.load("models/robust_model.pth"))
        robust.eval()
        has_robust = True
    except:
        print("Robust model not found, skipping Adv Training test.")
        has_robust = False

    squeezer = FeatureSqueezing(bit_depth=4)

    print("\n--- Defense Test ---")
    
    for i, (data, target) in enumerate(loader):
        if i >= 5: break
        
        data, target = data.to(device), target.to(device)
        label_name = labels[target.item()]
        
        # 1. Generate Attack on Baseline
        adv_data, _, _ = deepfool(baseline, data, device)
        
        # Predictions
        pred_clean = labels[baseline(data).argmax(dim=1)]
        pred_adv = labels[baseline(adv_data).argmax(dim=1)]
        
        if pred_clean != label_name: continue # Skip if baseline was wrong initially
        
        print(f"\nSample {i}: {label_name}")
        print(f"  [Baseline] Clean: {pred_clean} | Adv: {pred_adv}")
        
        # 2. Test Feature Squeezing
        squeezed_adv = squeezer(adv_data)
        pred_squeeze = labels[baseline(squeezed_adv).argmax(dim=1)]
        print(f"  [Defense: Squeezing] Prediction: {pred_squeeze}")
        
        # 3. Test Robust Model
        if has_robust:
            pred_robust = labels[robust(adv_data).argmax(dim=1)]
            print(f"  [Defense: Adv Train] Prediction: {pred_robust}")

if __name__ == "__main__":
    test_defenses()