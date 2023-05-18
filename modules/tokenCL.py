import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TokenCL(nn.Module):
    def __init__(self, hidden_size, cl_type, cl_temperature, emb_dim=128):
        super(TokenCL, self).__init__()
        self.emb_dim = emb_dim
        self.cl_type = cl_type
        self.cl_temperature = cl_temperature
        
        self.output_embedder_mu = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_size,
                      emb_dim)
        )
        self.output_embedder_sigma = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_size,
                      emb_dim)
        )

        # inter-intra weight
        # self.weight = nn.Parameter(torch.tensor([2., 1.]), requires_grad=True)

        
    def loss_kl(self, mu_i, sigma_i, mu_j, sigma_j, emb_dim):
        '''
        Calculates KL-divergence between two DIAGONAL Gaussians.
        Reference: https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians.
        Note: We calculated both directions of KL-divergence.
        '''
        sigma_ratio = sigma_j / sigma_i
        trace_fac = torch.sum(sigma_ratio, 1)
        log_det = torch.sum(torch.log(sigma_ratio + 1e-14), axis=1)
        mu_diff_sq = torch.sum((mu_i - mu_j) ** 2 / sigma_i, axis=1)
        ij_kl = 0.5 * (trace_fac + mu_diff_sq - emb_dim - log_det)
        sigma_ratio = sigma_i / sigma_j
        trace_fac = torch.sum(sigma_ratio, 1)
        log_det = torch.sum(torch.log(sigma_ratio + 1e-14), axis=1)
        mu_diff_sq = torch.sum((mu_j - mu_i) ** 2 / sigma_j, axis=1)
        ji_kl = 0.5 * (trace_fac + mu_diff_sq - emb_dim - log_det)
        kl_d = 0.5 * (ij_kl + ji_kl)
        return kl_d

    def loss_eu(self, mu_i, mu_j):
        mu_i = F.normalize(mu_i,  p=2, dim=-1)
        mu_j = F.normalize(mu_j,  p=2, dim=-1)
        # loss = ((mu_i - mu_j) ** 2).sum(dim=1)
        loss = F.mse_loss(mu_i, mu_j, reduction='none').sum(dim=1)
        return loss

    def loss_cosine(self, mu_i, mu_j):
        # mu_i = F.normalize(mu_i, p=2, dim=-1)
        # mu_j = F.normalize(mu_j, p=2, dim=-1)
        # loss = ((mu_i * mu_j).sum(dim=1))
        loss = F.cosine_similarity(mu_i, mu_j, dim=-1)
        return loss
    
    def loss_l1(self, mu_i, mu_j):
        mu_i = F.normalize(mu_i,  p=2, dim=-1)
        mu_j = F.normalize(mu_j,  p=2, dim=-1)
        # loss = ((mu_i - mu_j) ** 2).sum(dim=1)
        loss = F.l1_loss(mu_i, mu_j, reduction='none').sum(dim=1)
        return loss

    def loss_sml1(self, mu_i, mu_j):
        mu_i = F.normalize(mu_i,  p=2, dim=-1)
        mu_j = F.normalize(mu_j,  p=2, dim=-1)
        # loss = ((mu_i - mu_j) ** 2).sum(dim=1)
        loss = F.smooth_l1_loss(mu_i, mu_j, reduction='none', beta=0.5).sum(dim=1)
        return loss

    def nt_xent(self, loss, num, denom, temperature=1):
        "cross-entropy loss for CL"
        # TODO: grad temperature
        # loss = torch.exp(loss * temperature)
        loss = torch.exp(loss / temperature)
        cnts = torch.sum(num, dim = 1)
        loss_num = torch.sum(loss * num, dim = 1)
        loss_denom = torch.sum(loss * denom, dim = 1)
        # sanity check
        nonzero_indexes = torch.where(cnts > 0)
        loss_num, loss_denom, cnts = loss_num[nonzero_indexes], loss_denom[nonzero_indexes], cnts[nonzero_indexes]

        loss_final = -torch.log(loss_num) + torch.log(loss_denom) + torch.log(cnts)
        return loss_final

    def circleLoss(self, loss, pos_mask, neg_mask, margin=0.4, gamma=32.):
        """
        :param embeddings_a: (B, D)
        :param embeddings_b: (B, D)
        :return: The scalar loss
        """
        pos_cosine = loss * pos_mask.float()
        neg_cosine = loss * neg_mask.float()
        # Perturbed Circle Loss
        neg_loss = torch.sum(torch.exp(gamma * ((neg_cosine + margin) * (neg_cosine - margin))), dim=1)
        pos_loss = torch.sum(torch.exp(-gamma * ((1. + margin - pos_cosine) * (pos_cosine - 1. + margin))), dim=1)
        circle_loss = torch.mean(torch.log(1. + neg_loss * pos_loss))
        
        return circle_loss

    def calculate_kl_loss(self, original_embedding_mu, original_embedding_sigma, labels,
                          mask, consider_mutual_O=False, type_loss='kl'):
        non_padding_idxs = mask.view(-1) == 1
        non_padding_idxs = torch.where(non_padding_idxs == True)[0]

        # exclude padding
        non_padding_embedding_mu = original_embedding_mu.view(-1, self.emb_dim)[non_padding_idxs]
        non_padding_embedding_sigma = original_embedding_sigma.view(-1, self.emb_dim)[non_padding_idxs]
        non_padding_label = labels.view(-1)[non_padding_idxs]

        # do not consider tokens with 'O' classes.
        if not consider_mutual_O:
            filter_idxs = torch.where(non_padding_label > 0)[0]
            filter_embedding_mu = non_padding_embedding_mu[filter_idxs]
            filter_embedding_sigma = non_padding_embedding_sigma[filter_idxs]
            filter_labels = non_padding_label[filter_idxs]
        # consider entity-O, O-entity, O-O
        else:
            filter_embedding_mu = non_padding_embedding_mu
            filter_embedding_sigma = non_padding_embedding_sigma
            filter_labels = non_padding_label
        num_filter_samples = len(filter_labels)
        
        # repeat interleave(1, 1, 2, 2, ..., n, n)
        inter_embedding_mu = filter_embedding_mu.view(num_filter_samples, 1, self.emb_dim).expand(num_filter_samples, num_filter_samples, self.emb_dim).contiguous().view(num_filter_samples * num_filter_samples, -1)
        inter_embedding_sigma = filter_embedding_sigma.view(num_filter_samples, 1, self.emb_dim).expand(num_filter_samples, num_filter_samples, self.emb_dim).contiguous().view(num_filter_samples * num_filter_samples, -1)
        inter_labels = filter_labels.view(num_filter_samples, 1).expand(num_filter_samples, num_filter_samples).contiguous().view(num_filter_samples * num_filter_samples)

        # repeat tensor(1, 2, ..., n, 1, 2, ..., n, ...)
        repeat_embedding_mu = filter_embedding_mu.view(1, num_filter_samples, self.emb_dim).expand(num_filter_samples, num_filter_samples, self.emb_dim).contiguous().view(num_filter_samples * num_filter_samples, self.emb_dim)
        repeat_embedding_sigma = filter_embedding_sigma.view(1, num_filter_samples, self.emb_dim).expand(num_filter_samples, num_filter_samples, self.emb_dim).contiguous().view(num_filter_samples * num_filter_samples, self.emb_dim)
        repeat_labels = filter_labels.view(1, num_filter_samples).expand(num_filter_samples, num_filter_samples).contiguous().view(num_filter_samples * num_filter_samples)
        
        assert len(repeat_labels) == (num_filter_samples * num_filter_samples)

        # # create mask top-k selection
        # sim_score = F.cosine_similarity(inter_embedding_mu, repeat_embedding_mu, dim=-1).view(num_filter_samples, num_filter_samples)
        # top_k = torch.where(sim_score > 0.4)
        # neg_mask = torch.zeros((num_filter_samples, num_filter_samples), device=labels.device)
        # neg_mask[top_k[0], top_k[1]] = 1
        # neg_mask = neg_mask.view(-1)
        # denominator_mask = torch.all(inter_embedding_mu != repeat_embedding_mu, dim=-1).float() * torch.logical_or(neg_mask, (inter_labels==repeat_labels).float())
        # numerator_mask = torch.all(inter_embedding_mu != repeat_embedding_mu, dim=-1).float() * (inter_labels == repeat_labels).float()

        denominator_mask = torch.all(inter_embedding_mu != repeat_embedding_mu, dim=-1)
        numerator_mask = denominator_mask * (inter_labels == repeat_labels)
        # neg_mask = denominator_mask * (inter_labels != repeat_labels)
        
        # eq{4}, d(p, q) = KL
        if type_loss == 'kl':
            loss = -self.loss_kl(inter_embedding_mu, inter_embedding_sigma, 
                                repeat_embedding_mu, repeat_embedding_sigma, 
                                emb_dim=self.emb_dim)
        elif type_loss == 'euclidean':
            loss = -self.loss_eu(inter_embedding_mu, repeat_embedding_mu)
        elif type_loss == 'cosine':
            loss = self.loss_cosine(inter_embedding_mu, repeat_embedding_mu)
        elif type_loss == 'kl_div':
            loss = -0.5 * (F.kl_div(F.log_softmax(inter_embedding_mu, dim=-1), F.softmax(repeat_embedding_mu, dim=-1), reduction='none') + \
                F.kl_div(F.log_softmax(repeat_embedding_mu, dim=-1), F.softmax(inter_embedding_mu, dim=-1), reduction='none')).sum(dim=-1)
        elif type_loss == 'l1':
            loss = -self.loss_l1(inter_embedding_mu, repeat_embedding_mu)
        elif type_loss == 'smoothl1':
            loss = -self.loss_sml1(inter_embedding_mu, repeat_embedding_mu)
        loss = loss.view(num_filter_samples, num_filter_samples)

        # positive pair - margin
        # margin = numerator_mask.view(num_filter_samples, num_filter_samples) * 0.3
        # loss -= margin

        numerator_mask = numerator_mask.view(num_filter_samples, num_filter_samples)
        denominator_mask = denominator_mask.view(num_filter_samples, num_filter_samples)
        # neg_mask = neg_mask.view(num_filter_samples, num_filter_samples)
        
        # return self.circleLoss(loss, pos_mask=numerator_mask, neg_mask=neg_mask)
        # exp / sum(exp)
        return torch.mean(self.nt_xent(loss, numerator_mask.float(), denominator_mask.float(), temperature=self.cl_temperature))
    
    def forward(self, token_repr, labels, mask):
        original_embedding_mu = self.output_embedder_mu(token_repr)
        original_embedding_sigma = F.elu(self.output_embedder_sigma(token_repr)) + 1 + 1e-14
        
        return self.calculate_kl_loss(original_embedding_mu, original_embedding_sigma, 
                                    labels, mask, type_loss=self.cl_type)

    def inter_intra_loss(self, original_embedding_mu, original_embedding_sigma, labels, mask, consider_mutual_O=False, type_loss='kl', bound_ids=None):
        def inter_intra_nt_xent(loss, mask_a, mask_b, mask_c, mask_bc, temperature=1):
            "cross-entropy loss for CL"
            loss = torch.exp(loss / temperature)
            # cnts = torch.sum(mask_a, dim = 1)
            loss_a = torch.sum(loss * mask_a, dim=1)
            loss_b = torch.sum(loss * mask_b, dim=1)
            loss_c = torch.sum(loss * mask_c, dim=1)
            loss_bc = torch.sum(loss * mask_bc, dim=1)
            # sanity check
            # nonzero_indexes = torch.where(cnts > 0)
            # loss_num, loss_denom, cnts = loss_num[nonzero_indexes], loss_denom[nonzero_indexes], cnts[nonzero_indexes]

            loss_final = -torch.log(loss_a / loss_b) + torch.log(loss_a / loss_b + loss_c / loss_bc)
            return loss_final
        non_padding_idxs = mask.view(-1) == 1
        non_padding_idxs = torch.where(non_padding_idxs == True)[0]

        # exclude padding
        non_padding_embedding_mu = original_embedding_mu.view(-1, self.emb_dim)[non_padding_idxs]
        non_padding_embedding_sigma = original_embedding_sigma.view(-1, self.emb_dim)[non_padding_idxs]
        non_padding_label = labels.view(-1)[non_padding_idxs]
        non_padding_bounds = bound_ids.view(-1)[non_padding_idxs]

        # do not consider tokens with 'O' classes.
        filter_idxs = torch.where(non_padding_label > 0)[0]
        filter_embedding_mu = non_padding_embedding_mu[filter_idxs]
        filter_embedding_sigma = non_padding_embedding_sigma[filter_idxs]
        filter_labels = non_padding_label[filter_idxs]
        filter_bounds = non_padding_bounds[filter_idxs]
        
        num_filter_samples = len(filter_labels)

        # repeat interleave(1, 1, 2, 2, ..., n, n)
        inter_embedding_mu = filter_embedding_mu.view(num_filter_samples, 1, self.emb_dim).expand(num_filter_samples, num_filter_samples, self.emb_dim).contiguous().view(num_filter_samples * num_filter_samples, -1)
        inter_embedding_sigma = filter_embedding_sigma.view(num_filter_samples, 1, self.emb_dim).expand(num_filter_samples, num_filter_samples, self.emb_dim).contiguous().view(num_filter_samples * num_filter_samples, -1)
        inter_labels = filter_labels.view(num_filter_samples, 1).expand(num_filter_samples, num_filter_samples).contiguous().view(num_filter_samples * num_filter_samples)
        inter_bounds = filter_bounds.view(num_filter_samples, 1).expand(num_filter_samples, num_filter_samples).contiguous().view(num_filter_samples * num_filter_samples)

        # repeat tensor(1, 2, ..., n, 1, 2, ..., n, ...)
        repeat_embedding_mu = filter_embedding_mu.view(1, num_filter_samples, self.emb_dim).expand(num_filter_samples, num_filter_samples, self.emb_dim).contiguous().view(num_filter_samples * num_filter_samples, self.emb_dim)
        repeat_embedding_sigma = filter_embedding_sigma.view(1, num_filter_samples, self.emb_dim).expand(num_filter_samples, num_filter_samples, self.emb_dim).contiguous().view(num_filter_samples * num_filter_samples, self.emb_dim)
        repeat_labels = filter_labels.view(1, num_filter_samples).expand(num_filter_samples, num_filter_samples).contiguous().view(num_filter_samples * num_filter_samples)
        repeat_bounds = filter_bounds.view(1, num_filter_samples).expand(num_filter_samples, num_filter_samples).contiguous().view(num_filter_samples * num_filter_samples)

        assert len(repeat_labels) == (num_filter_samples * num_filter_samples)

        denominator_mask = torch.all(inter_embedding_mu != repeat_embedding_mu, dim=-1).float()
        numerator_mask = denominator_mask * (inter_labels == repeat_labels).float()

        # intra mask (1, 1, 1, 2, 3, ...) -> (B-1, I-1, I-1, B-2, B-3, ...)
        intra_mask = inter_bounds == repeat_bounds
        inter_mask = inter_labels == repeat_labels
        intra_denominator_mask = (torch.all(inter_embedding_mu != repeat_embedding_mu, dim=-1).float() * inter_mask.float())
        intra_numerator_mask = (intra_denominator_mask * intra_mask.float())
    
        # eq{4}, d(p, q) = KL
        if type_loss == 'kl':
            loss = -self.loss_kl(inter_embedding_mu, inter_embedding_sigma, 
                                repeat_embedding_mu, repeat_embedding_sigma, 
                                emb_dim=self.emb_dim)
        elif type_loss == 'euclidean':
            loss = -self.loss_eu(inter_embedding_mu, repeat_embedding_mu)
        elif type_loss == 'cosine':
            loss = self.loss_cosine(inter_embedding_mu, repeat_embedding_mu)
        loss = loss.view(num_filter_samples, num_filter_samples)

        numerator_mask = numerator_mask.view(num_filter_samples, num_filter_samples)
        denominator_mask = denominator_mask.view(num_filter_samples, num_filter_samples)
        intra_numerator_mask = intra_numerator_mask.view(num_filter_samples, num_filter_samples)
        intra_denominator_mask = intra_denominator_mask.view(num_filter_samples, num_filter_samples)

        return torch.mean(inter_intra_nt_xent(loss, intra_mask.view(num_filter_samples, num_filter_samples), 
                                              inter_mask.view(num_filter_samples, num_filter_samples),
                                              (~(inter_labels == repeat_labels)).float().view(num_filter_samples, num_filter_samples), 
                                              torch.ones((num_filter_samples, num_filter_samples), device=labels.device),
                                              temperature=self.cl_temperature))
        
        # # VAE sampling
        # vae_kl_loss, vae_repr = self.vae(token_repr, step, self.k, self.x0)
        # return self.calculate_vae_loss(vae_repr, labels, mask), vae_kl_loss

    def calculate_vae_loss(self, original_repr, labels, mask, type_loss='kl'):
        non_padding_idxs = mask.view(-1) == 1
        non_padding_idxs = torch.where(non_padding_idxs == True)[0]

        # exclude padding
        non_padding_embedding = original_repr.view(-1, self.emb_dim)[non_padding_idxs]
        non_padding_label = labels.view(-1)[non_padding_idxs]

        # consider tokens with 'O' classes with other "O" classes tokens.
        # filter_idxs = torch.where(non_padding_label > 0)[0]
        filter_embedding = non_padding_embedding
        filter_labels = non_padding_label
        num_filter_samples = len(filter_labels)

        # repeat interleave(1, 1, 2, 2, ..., n, n)
        filter_embedding = torch.repeat_interleave(filter_embedding, len(non_padding_embedding), dim=0)
        filter_labels = torch.repeat_interleave(filter_labels, len(non_padding_label), dim=0)

        # repeat tensor(1, 2, ..., n, 1, 2, ..., n, ...)
        repeat_embedding = non_padding_embedding.repeat(num_filter_samples, 1)
        repeat_labels = non_padding_label.repeat(num_filter_samples)

        assert len(repeat_labels) == (num_filter_samples * num_filter_samples)

        # create mask
        # p != q
        denominator_mask = torch.all(filter_embedding != repeat_embedding, dim=-1).float()
        numerator_mask = denominator_mask * (filter_labels == repeat_labels).float()
        
        # eq{4}, d(p, q) = KL
        loss = -0.5 * (F.kl_div(F.log_softmax(filter_embedding, dim=-1), F.softmax(repeat_embedding, dim=-1), reduction='none').sum(dim=1) + 
                        F.kl_div(F.log_softmax(repeat_embedding, dim=-1), F.softmax(filter_embedding, dim=-1), reduction='none').sum(dim=1))

        loss = loss.view(num_filter_samples, num_filter_samples)
        numerator_mask = numerator_mask.view(num_filter_samples, num_filter_samples)
        denominator_mask = denominator_mask.view(num_filter_samples, num_filter_samples)

        return torch.mean(self.nt_xent(loss, numerator_mask, denominator_mask, temperature=self.cl_temperature))
