"""Selector training on LLaVA-Video-178K pseudo-labels.

The pseudo-labels score groups of temporal segments of a video for one
question. Segments are the level-0 clusters of temporally weighted FINCH on
SigLIP frame features. Each training input is the BLIP fused question tokens
averaged over the frames of the selected segments, and the target is the
pseudo-label of that segment group.
"""
