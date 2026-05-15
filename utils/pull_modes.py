import torch
import torch.nn.functional as F


def compute_beta_from_consensus(
    base_feat,
    aug_feats,
    beta_clean=0.45,
    beta_adv=2.25,
    gamma_attack=0.9,
    gamma_trust=1.0,
    eps=1e-8,
    return_stats=False,
):
    if base_feat.ndim == 2:
        base_feat = base_feat.squeeze(0)
    if aug_feats.ndim == 1:
        aug_feats = aug_feats.unsqueeze(0)

    base_feat = F.normalize(base_feat.view(1, -1).float(), dim=-1).squeeze(0)
    aug_feats = F.normalize(aug_feats.float(), dim=-1)

    cos_sim = torch.matmul(aug_feats, base_feat)
    outlierness = (1.0 - cos_sim).mean()
    attack_score = torch.clamp(outlierness, 0.0, 1.0)

    if aug_feats.size(0) > 1:
        sim_mat = aug_feats @ aug_feats.T
        eye = torch.eye(sim_mat.size(0), device=sim_mat.device, dtype=torch.bool)
        pair_sims = sim_mat.masked_select(~eye)
        reliability = pair_sims.mean()
        trust_score = torch.clamp((reliability + 1.0) * 0.5, 0.0, 1.0)
    else:
        reliability = torch.tensor(1.0, device=aug_feats.device)
        trust_score = torch.tensor(1.0, device=aug_feats.device)

    correction_score = (
        torch.pow(attack_score, gamma_attack) *
        torch.pow(trust_score, gamma_trust)
    )
    correction_score = torch.clamp(correction_score, 0.0, 1.0)

    beta = beta_clean + (beta_adv - beta_clean) * correction_score
    beta = float(beta.item())

    if return_stats:
        return beta, {
            "attack_score": float(attack_score.item()),
            "trust_score": float(trust_score.item()),
            "outlierness": float(outlierness.item()),
            "reliability_raw": float(reliability.item()),
            "correction_score": float(correction_score.item()),
        }

    return beta


def _build_anchor(aug_feats):
    return F.normalize(aug_feats.mean(dim=0, keepdim=True), dim=-1).squeeze(0)



def _geodesic_push(base_feat, anchor_feat, beta, eps=1e-8):
    x = F.normalize(base_feat.view(1, -1).float(), dim=-1).squeeze(0)
    a = F.normalize(anchor_feat.view(1, -1).float(), dim=-1).squeeze(0)

    cos_theta = torch.clamp(torch.dot(x, a), -1.0 + eps, 1.0 - eps)
    theta = torch.arccos(cos_theta)

    if float(theta.item()) < 1e-6:
        return x, {
            "geodesic_theta": float(theta.item()),
            "geodesic_step": 0.0,
        }, torch.zeros_like(x)

    tangent_vec = a - torch.dot(x, a) * x
    tangent_norm = torch.norm(tangent_vec).clamp_min(eps)
    u = tangent_vec / tangent_norm
    
    step = beta * theta
    push_feat = torch.cos(step) * x + torch.sin(step) * u
    push_feat = F.normalize(push_feat.view(1, -1), dim=-1).squeeze(0)

    return push_feat, {
        "geodesic_theta": float(theta.item()),
        "geodesic_step": float(step.item()),
    }, u


def step_feature(base_feat, aug_feats, step, return_stats=False):
    anchor_feat = _build_anchor(aug_feats)
    push_feat, geo_stats, _ = _geodesic_push(base_feat, anchor_feat, step)
    filtered_feat = F.normalize(push_feat.unsqueeze(0), dim=-1)
    if return_stats:
        stats = {
            "beta": step,
            **geo_stats,
        }
        return filtered_feat, stats

    return filtered_feat ## if use u to predict max label. return u


def build_tangent_filtered_feature(
    base_feat,
    aug_feats,
    beta_clean=0.45,
    beta_adv=2.4,
    eps=1e-8,
    return_stats=False,
):
    if base_feat.ndim == 2:
        base_feat = base_feat.squeeze(0)
    if aug_feats.ndim == 1:
        aug_feats = aug_feats.unsqueeze(0)

    anchor_feat = _build_anchor(aug_feats)
    beta, beta_stats = compute_beta_from_consensus(
        base_feat=base_feat,
        aug_feats=aug_feats,
        beta_clean=beta_clean,
        beta_adv=beta_adv,
        eps=eps,
        return_stats=True,
    )


    push_feat, geo_stats, _ = _geodesic_push(
        base_feat=base_feat,
        anchor_feat=anchor_feat,
        beta=beta,
        eps=eps,
    )
    filtered_feat = F.normalize(push_feat.unsqueeze(0), dim=-1)
    
    if return_stats:
        stats = {
            "beta": beta,
            **beta_stats,
            **geo_stats,
        }
        return filtered_feat, stats

    return filtered_feat
