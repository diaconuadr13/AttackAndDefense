import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import SubsetSC, collate_fn, get_labels
from models.simple_cnn import SimpleAudioCNN
from attacks.deepfool import deepfool
from defenses.squeezing import FeatureSqueezing

def evaluate_defenses(num_samples=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device}")
    
    test_set = SubsetSC("testing")
    loader = DataLoader(test_set, batch_size=1, shuffle=True, collate_fn=collate_fn)
    
    # Load Baseline
    baseline = SimpleAudioCNN(n_classes=35).to(device)
    try:
        baseline.load_state_dict(torch.load("models/baseline_model.pth"))
    except:
        print("Baseline model not found.")
        return
    baseline.eval()
    
    # Load Robust Model
    robust = SimpleAudioCNN(n_classes=35).to(device)
    try:
        robust.load_state_dict(torch.load("models/robust_model.pth"))
        robust.eval()
        has_robust = True
        print("Robust model loaded.")
    except:
        print("Robust model NOT found. Run train_adv.py first.")
        has_robust = False

    # Initialize Defense
    squeezer = FeatureSqueezing(bit_depth=5)

    total_attacks = 0
    squeezing_saved = 0
    robust_saved = 0
    
    print(f"\n--- Running Defense Evaluation on {num_samples} successful attacks ---")
    
    pbar = tqdm(total=num_samples)
    
    # Iterate through data until we have generated 'num_samples' successful attacks
    for i, (data, target) in enumerate(loader):
        if total_attacks >= num_samples:
            break
        
        data, target = data.to(device), target.to(device)
        label_idx = target.item()
        
        # 1. Check if Baseline is correct
        pred_clean = baseline(data).argmax(dim=1).item()
        if pred_clean != label_idx: continue 
        
        # 2. Generate Attack (DeepFool)
        adv_data, _, pred_adv_idx = deepfool(baseline, data, device, max_iter=50)
        
        # Only proceed if attack was successful (model was fooled)
        if pred_adv_idx == label_idx: continue
        
        total_attacks += 1
        pbar.update(1)
        
        # 3. Test Feature Squeezing
        squeezed_input = squeezer(adv_data)
        pred_squeeze_idx = baseline(squeezed_input).argmax(dim=1).item()
        
        if pred_squeeze_idx == label_idx:
            squeezing_saved += 1
        
        # 4. Test Adversarial Training (Robust Model)
        if has_robust:
            pred_robust_idx = robust(adv_data).argmax(dim=1).item()
            if pred_robust_idx == label_idx:
                robust_saved += 1
                
    pbar.close()

    print("\n" + "="*40)
    print(f"Total Successful Attacks Generated: {total_attacks}")
    print("-" * 40)
    print(f"Feature Squeezing Defense Accuracy: {squeezing_saved / total_attacks * 100:.2f}%")
    if has_robust:
        print(f"Adversarial Training Defense Accuracy: {robust_saved / total_attacks * 100:.2f}%")
    print("="*40)

if __name__ == "__main__":
    # Evaluate defenses on 100 successful adversarial examples
    evaluate_defenses(num_samples=100)