r""" Provides functions that builds/manipulates correlation tensors """
import torch


class Similarity:

    @classmethod
    def multilayer_similarity(cls, query_feats, support_feats, stack_ids):
        eps = 1e-5

        sims = []
        for idx, (query_feat, support_feat) in enumerate(zip(query_feats, support_feats)):
            bsz, ch, hb, wb = support_feat.size()
            support_feat = support_feat.view(bsz, ch, -1)
            support_feat = support_feat / (support_feat.norm(dim=1, p=2, keepdim=True) + eps)

            bsz, ch, ha, wa = query_feat.size()
            query_feat = query_feat.view(bsz, ch, -1)
            query_feat = (query_feat / (query_feat.norm(dim=1, p=2, keepdim=True) + eps))

            sim = torch.bmm(query_feat.transpose(1, 2), support_feat).view(bsz, ha, wa, hb, wb)
            sim = sim.clamp(min=0)
            sims.append(sim)

        sims_l4 = torch.stack(sims[-stack_ids[0]:]).transpose(0, 1).contiguous()
        sims_l3 = torch.stack(sims[-stack_ids[1]:-stack_ids[0]]).transpose(0, 1).contiguous()
        sims_l2 = torch.stack(sims[-stack_ids[2]:-stack_ids[1]]).transpose(0, 1).contiguous()

        return [sims_l4, sims_l3, sims_l2]
