import torch

def batch_tensors(*ts):
    # Temporary fix for different sequence lengths: trim to the shortest sequence.
    # In practice, this drops the last day of data at the end of the year on leap years.
    # Should be removed when a proper solution is implemented upstream.
    if all([isinstance(t, torch.Tensor) for t in ts]) and all([t.ndim == 1 for t in ts]):
        min_len = min([t.size(0) for t in ts]) # Get minimum sequence length
        ts = [t[:min_len] for t in ts] # Trim all sequences to the same length
        return torch.cat([t.unsqueeze(0) for t in ts], dim=0) # Stack all sequences
    else:
        return torch.cat([t.unsqueeze(0) for t in ts], dim=0)
