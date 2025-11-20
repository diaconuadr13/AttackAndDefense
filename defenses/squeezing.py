import torch

class FeatureSqueezing:
    def __init__(self, bit_depth=4):
        """
        Feature Squeezing using Bit Depth Reduction.
        Reduces the precision of the input tensor features.
        bit_depth: Number of bits to keep (e.g., 4, 5, 8). Lower = stronger squeezing.
        """
        self.bit_depth = bit_depth
        self.max_val = 2**bit_depth - 1

    def __call__(self, x):
        # x: Input tensor (MFCC features)
        
        # 1. Normalize to [0, 1] per sample to handle varying amplitudes
        x_min = x.min()
        x_max = x.max()
        
        # Avoid division by zero
        if x_max - x_min == 0:
            return x
            
        x_norm = (x - x_min) / (x_max - x_min)
        
        # 2. Quantize (Squeeze)
        # Map 0-1 to 0-max_val, round to nearest integer, map back
        x_quant = torch.round(x_norm * self.max_val) / self.max_val
        
        # 3. Denormalize back to original range
        x_squeezed = x_quant * (x_max - x_min) + x_min
        
        return x_squeezed