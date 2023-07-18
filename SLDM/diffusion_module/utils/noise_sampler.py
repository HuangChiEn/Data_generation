from sklearn.mixture import GaussianMixture

def get_noise_sampler(sample_type='gau'):
    if sample_type == 'gau':
        sampler = lambda latnt_sz: torch.randn_like(latnt_sz)
    elif sample_type == 'gau_offset':
        sampler = lambda latnt_sz: torch.randn_like(latnt_sz) + (torch.randn_like(latnt_sz))
        ...
    elif sample_type == 'gmm':
        ...
    else:
        ...
    return 

if __name__ == "__main__":
    ...