import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import SubsetSC, collate_fn, get_labels
from models.simple_cnn import SimpleAudioCNN
from attacks.deepfool import deepfool
from attacks.cw import cw_l2_attack
from defenses.squeezing import FeatureSqueezing

def run_defense_eval(attack_name, attack_fn, model, robust_model, loader, device, num_samples, squeezer):
    print(f"\n--- Running Defense Evaluation against {attack_name} ---")
    
    total_attacks = 0
    squeezing_saved = 0
    robust_saved = 0
    
    pbar = tqdm(total=num_samples)
    
    for i, (data, target) in enumerate(loader):
        if total_attacks >= num_samples:
            break
        
        data, target = data.to(device), target.to(device)
        label_idx = target.item()
        
        # Check if Baseline is correct
        pred_clean = model(data).argmax(dim=1).item()
        if pred_clean != label_idx: continue 
        
        # Generate Attack
        if attack_name == "DeepFool":
            adv_data, _, pred_adv_idx = attack_fn(model, data, device, max_iter=50)
        elif attack_name == "C&W":
            # C&W returns just the adversarial example
            adv_data = attack_fn(model, data, target, device, c=10, max_iter=100, binary_search_steps=3)
            pred_adv_idx = model(adv_data).argmax(dim=1).item()
            
        # Only proceed if attack was successful (model was fooled)
        if pred_adv_idx == label_idx: continue
        
        total_attacks += 1
        pbar.update(1)
        
        # Test Feature Squeezing
        squeezed_input = squeezer(adv_data)
        pred_squeeze_idx = model(squeezed_input).argmax(dim=1).item()
        
        if pred_squeeze_idx == label_idx:
            squeezing_saved += 1
        
        # Test Adversarial Training (Robust Model)
        if robust_model:
            pred_robust_idx = robust_model(adv_data).argmax(dim=1).item()
            if pred_robust_idx == label_idx:
                robust_saved += 1
                
    pbar.close()
    
    print(f"\nResults for {attack_name}:")
    print(f"  Feature Squeezing Accuracy: {squeezing_saved / total_attacks * 100:.2f}%")
    if robust_model:
        print(f"  Adversarial Training Accuracy: {robust_saved / total_attacks * 100:.2f}%")

def evaluate_defenses(num_samples=50):
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
        robust.load_state_dict(torch.load("models/robust_model_2.pth"))
        robust.eval()
        print("Robust model loaded.")
    except:
        print("Robust model NOT found. Run train_adv.py first.")
        robust = None

    # Initialize Defense
    squeezer = FeatureSqueezing(bit_depth=5)

    # Run DeepFool eval
    run_defense_eval("DeepFool", deepfool, baseline, robust, loader, device, num_samples, squeezer)
    
    # Run C&W eval
    # Re-create loader to ensure fresh samples or shuffle
    loader = DataLoader(test_set, batch_size=1, shuffle=True, collate_fn=collate_fn)
    run_defense_eval("C&W", cw_l2_attack, baseline, robust, loader, device, num_samples, squeezer)

if __name__ == "__main__":
    evaluate_defenses(num_samples=50)