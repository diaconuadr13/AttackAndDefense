import torch
import torch.optim as optim

def cw_l2_attack(model, x, device, target_class=None, c=1, kappa=0, max_iter=1000, learning_rate=0.01):
    """
    Carlini & Wagner L2 Attack.
    Optimizes: min ||x_adv - x||^2 + c * f(x_adv)
    """
    model.eval()
    
    # w is the variable we optimize. x_adv = 0.5 * (tanh(w) + 1)
    # However, since audio features aren't strictly bounded like pixels [0,1],
    # we will optimize x_adv directly or assume a bound if your MFCCs are normalized.
    # For simplicity with MFCCs (unbounded), we optimize delta directly.
    
    delta = torch.zeros_like(x, requires_grad=True, device=device)
    optimizer = optim.Adam([delta], lr=learning_rate)

    for step in range(max_iter):
        x_adv = x + delta
        output = model(x_adv)
        
        # L2 distance loss
        l2_loss = torch.sum(delta ** 2)

        # f-function (Attack Loss)
        # We want the score of the target class to be higher than the rest
        # or (untargeted) the score of the correct class to be lower than the runner-up.
        
        current_label = output.argmax(dim=1)
        correct_logit = output[0, current_label]
        
        # Find max logit other than correct one
        # (Simple logic for untargeted attack)
        sorted_logits, _ = torch.sort(output, dim=1, descending=True)
        
        if sorted_logits[0, 0] == correct_logit:
            # If correct class is top, take the second highest
            max_other = sorted_logits[0, 1]
        else:
            # If correct class is not top, take the highest
            max_other = sorted_logits[0, 0]

        # For untargeted: we want max_other > correct_logit
        # f(x) = max(0, correct_logit - max_other + kappa)
        f_loss = torch.clamp(correct_logit - max_other + kappa, min=0)

        loss = l2_loss + c * f_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Stop if we fooled the model (f_loss is 0) and continue to minimize L2
        if f_loss.item() == 0 and step > max_iter // 2:
             # Early stopping can be added here if strict L2 minimization isn't priority
             pass

    return (x + delta).detach()