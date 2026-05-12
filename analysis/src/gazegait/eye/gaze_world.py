"""Compute gaze-in-world from gaze-in-frame + head orientation.

Raw Neon gaze is expressed relative to the scene camera, which is rigidly
attached to the head. Every footstep wobbles the head, the scene camera moves
with it, and gaze-in-frame therefore oscillates even when the eye is steadily
fixating a real-world point.

Rotating the gaze unit vector by the head orientation (quaternion or RPY)
yields the world-stabilized gaze, which is what fixation / saccade detection
must consume.
"""

from __future__ import annotations


def gaze_to_world(t_utc_ns_gaze, gaze_xy_norm, t_utc_ns_head, head_quat_wxyz):
    """Return (t_utc_ns, gaze_world_unit_vec) sampled at the gaze timeline."""
    raise NotImplementedError
