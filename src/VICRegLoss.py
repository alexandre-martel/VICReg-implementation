import torch
import torch.nn.functional as F

def vicreg_loss(z1, z2, sim_coeff=25.0, std_coeff=25.0, cov_coeff=1.0, gamma=1, espilon=1e-4):
    """
    Compute the VICReg loss between two sets of projections z1 and z2. The coefficients are those used in the original VICReg paper.

    Args:
        z1: Projections from view 1, shape (batch_size, projection_dim)
        z2: Projections from view 2, shape (batch_size, projection_dim)
        sim_coeff: Coefficient for the invariance loss term
        std_coeff: Coefficient for the variance loss term
        cov_coeff: Coefficient for the covariance loss term
    """
    # We have to compute the 3 losses here: invariance, variance, covariance and then add them with their respective coefficients.

    # INVARIANCE LOSS
    invariance_loss = torch.mean((z1 - z2) ** 2)

    # VARIANCE LOSS
    std1 = regularized_std(z1, espilon)
    std1_loss = torch.mean(F.relu(gamma - std1))

    std2 = regularized_std(z2, espilon)
    std2_loss = torch.mean(F.relu(gamma - std2))

    variance_loss = (std1_loss + std2_loss) / 2

    # COVARIANCE LOSS
    cov1 = covariance(z1)
    cov2 = covariance(z2)

    cov1_loss = (cov1.pow(2).sum() - cov1.diagonal().pow(2).sum()) / z1.size(1)
    cov2_loss = (cov2.pow(2).sum() - cov2.diagonal  ().pow(2).sum()) / z2.size(1)

    covariance_loss = (cov1_loss + cov2_loss) / 2

    # TOTAL VICREG LOSS
    total_loss = (sim_coeff * invariance_loss) + (std_coeff * variance_loss) + (cov_coeff * covariance_loss)
    return total_loss

def regularized_std(z, espilon=1e-4):
    std = torch.sqrt(z.var(dim=0) + espilon)
    return std

def covariance(z):
    N, _ = z.size()
    z_unbiased = z - z.mean(dim=0)
    cov = (z_unbiased.T @ z_unbiased) / (N - 1)
    return cov