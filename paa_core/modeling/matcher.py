# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import numpy as np
import torch.distributions.normal as tdn


class Matcher(object):
    """
    This class assigns to each predicted "element" (e.g., a box) a ground-truth
    element. Each predicted element will have exactly zero or one matches; each
    ground-truth element may be assigned to zero or more predicted elements.

    Matching is based on the MxN match_quality_matrix, that characterizes how well
    each (ground-truth, predicted)-pair match. For example, if the elements are
    boxes, the matrix may contain box IoU overlap values.

    The matcher returns a tensor of size N containing the index of the ground-truth
    element m that matches to prediction n. If there is no match, a negative value
    is returned.
    """

    BELOW_LOW_THRESHOLD = -1
    BETWEEN_THRESHOLDS = -2

    def __init__(self, high_threshold, low_threshold, allow_low_quality_matches=False):
        """
        Args:
            high_threshold (float): quality values greater than or equal to
                this value are candidate matches.
            low_threshold (float): a lower quality threshold used to stratify
                matches into three levels:
                1) matches >= high_threshold
                2) BETWEEN_THRESHOLDS matches in [low_threshold, high_threshold)
                3) BELOW_LOW_THRESHOLD matches in [0, low_threshold)
            allow_low_quality_matches (bool): if True, produce additional matches
                for predictions that have only low-quality match candidates. See
                set_low_quality_matches_ for more details.
        """
        assert low_threshold <= high_threshold
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.allow_low_quality_matches = allow_low_quality_matches

    def __call__(self, match_quality_matrix):
        """
        Args:
            match_quality_matrix (Tensor[float]): an MxN tensor, containing the
            pairwise quality between M ground-truth elements and N predicted elements.

        Returns:
            matches (Tensor[int64]): an N tensor where N[i] is a matched gt in
            [0, M - 1] or a negative value indicating that prediction i could not
            be matched.
        """
        if match_quality_matrix.numel() == 0:
            # empty targets or proposals not supported during training
            if match_quality_matrix.shape[0] == 0:
                raise ValueError(
                    "No ground-truth boxes available for one of the images "
                    "during training")
            else:
                raise ValueError(
                    "No proposal boxes available for one of the images "
                    "during training")

        # match_quality_matrix is M (gt) x N (predicted)
        # Max over gt elements (dim 0) to find best gt candidate for each prediction
        matched_vals, matches = match_quality_matrix.max(dim=0)
        if self.allow_low_quality_matches:
            all_matches = matches.clone()

        # Assign candidate matches with low quality to negative (unassigned) values
        below_low_threshold = matched_vals < self.low_threshold
        between_thresholds = (matched_vals >= self.low_threshold) & (
            matched_vals < self.high_threshold
        )
        matches[below_low_threshold] = Matcher.BELOW_LOW_THRESHOLD
        matches[between_thresholds] = Matcher.BETWEEN_THRESHOLDS

        if self.allow_low_quality_matches:
            self.set_low_quality_matches_(matches, all_matches, match_quality_matrix)

        return matches

    def set_low_quality_matches_(self, matches, all_matches, match_quality_matrix):
        """
        Produce additional matches for predictions that have only low-quality matches.
        Specifically, for each ground-truth find the set of predictions that have
        maximum overlap with it (including ties); for each prediction in that set, if
        it is unmatched, then match it to the ground-truth with which it has the highest
        quality value.
        """
        # For each gt, find the prediction with which it has highest quality
        highest_quality_foreach_gt, _ = match_quality_matrix.max(dim=1)
        # Find highest quality match available, even if it is low, including ties
        gt_pred_pairs_of_highest_quality = torch.nonzero(
            match_quality_matrix == highest_quality_foreach_gt[:, None],
            as_tuple=False
        )
        # Example gt_pred_pairs_of_highest_quality:
        #   tensor([[    0, 39796],
        #           [    1, 32055],
        #           [    1, 32070],
        #           [    2, 39190],
        #           [    2, 40255],
        #           [    3, 40390],
        #           [    3, 41455],
        #           [    4, 45470],
        #           [    5, 45325],
        #           [    5, 46390]])
        # Each row is a (gt index, prediction index)
        # Note how gt items 1, 2, 3, and 5 each have two ties

        pred_inds_to_update = gt_pred_pairs_of_highest_quality[:, 1]
        matches[pred_inds_to_update] = all_matches[pred_inds_to_update]

class CRPMatcher(object):

    def __init__(self, crp_alpha):
        self.crp_alpha = crp_alpha

    BELOW_LOW_THRESHOLD = -1
    BETWEEN_THRESHOLDS = -2

    def CRP_style_duplicate_solver(self, quality_matrix):
        """
        1. assign bin per each object
        2. for each candidate anchors, do crp process
        """
        iou_matrix = quality_matrix.clone().detach()

        N = quality_matrix.shape[0]
        device = quality_matrix.device
        quality_matrix = quality_matrix.t()
        #quality_matrix -= (quality_matrix.min(dim=0)[0].unsqueeze(0) - 1e-6).clamp(min=0)
        #quality_matrix /= quality_matrix.max(dim=0)[0].unsqueeze(0)
        M = len(quality_matrix) / N

        iou_bin = torch.ones(N,device=device)
        iou_mean = torch.zeros_like(iou_bin)
        idx = torch.zeros(len(quality_matrix), device=device).long()

        for i in range(len(quality_matrix) // N + int((len(quality_matrix) % N) > 0)):
            P_x_i = quality_matrix[i*N:(i+1)*N]
            #assert P_x_i.shape[0] == N
            Z_k = 1 / iou_bin
            #Z_k = (1 + ((iou_bin / M).clamp(max=0.99) * np.pi).cos()) / 2

            P_xz_i = P_x_i * Z_k.unsqueeze(0)
            curr_k = P_xz_i.argmax(dim=1)
            curr_unique, curr_count = curr_k.unique(return_counts=True)
            for j_unique, j_count in zip(curr_unique, curr_count):
                iou_bin[j_unique] += j_count
            idx[i*N:(i+1)*N] = curr_k
            
            for k_idx ,k in enumerate(curr_k):
                iou_mean[k] += iou_matrix[k, i*N + k_idx]

        iou_mean = (iou_mean / (iou_bin - 1)).mean()
        
        assert len(idx.unique()) == N

        return idx, {"iou_mean": iou_mean}

    def __call__(self, match_quality_matrix):
        """
        Args:
            match_quality_matrix (Tensor[float]): an MxN tensor, containing the
            pairwise quality between M ground-truth elements and N predicted elements.

        Returns:
            matches (Tensor[int64]): an N tensor where N[i] is a matched gt in
            [0, M - 1] or a negative value indicating that prediction i could not
            be matched.
        """
        if match_quality_matrix.numel() == 0:
            # empty targets or proposals not supported during training
            if match_quality_matrix.shape[0] == 0:
                raise ValueError(
                    "No ground-truth boxes available for one of the images "
                    "during training")
            else:
                raise ValueError(
                    "No proposal boxes available for one of the images "
                    "during training")

        A = match_quality_matrix.shape[1]
        device = match_quality_matrix.device
        _, topk_anchor_matches = match_quality_matrix.topk(100, dim=1)
        topk_anchor_matches = topk_anchor_matches.flatten().unique()
        topk_match_quality_matrix = match_quality_matrix[:,topk_anchor_matches]
        topk_matched_object_idxs, CRP_log = self.CRP_style_duplicate_solver(topk_match_quality_matrix)

        #matched_vals = torch.zeros(A, device=device)
        #matched_vals[topk_anchor_matches] = match_quality_matrix[topk_matched_object_idxs, topk_anchor_matches]

        matches = torch.ones(A, device=device).long() * CRPMatcher.BELOW_LOW_THRESHOLD
        matches[topk_anchor_matches] = topk_matched_object_idxs

        return matches, CRP_log