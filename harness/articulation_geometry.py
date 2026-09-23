"""Current RGB-D planar-joint search hints, never executable hinge poses.

Two nonparallel faces suggest a seam, but do NOT establish articulation or
object identity. The visual model must still identify the actual joint. No
task names, asset geometry, simulator state or historical coordinates.
"""
import numpy as np


def planar_joint_search(cloud, valid, box, table):
    xyz = np.asarray(cloud, float)
    mask = np.asarray(valid, bool).copy()
    if xyz.shape != mask.shape + (3,):
        return None
    h, w = mask.shape
    box = np.asarray(box, float)
    if (box.shape != (4,) or not np.isfinite(box).all() or np.any(box < 0)
            or np.any(box > 1000) or np.any(box[2:] <= box[:2])):
        return None
    x0, y0, x1, y1 = np.rint(box * [w-1, h-1, w-1, h-1] / 1000).astype(int)
    region = np.zeros_like(mask)
    region[y0:y1+1, x0:x1+1] = True
    mask &= region & np.isfinite(xyz).all(2)
    mask &= (xyz[:, :, 2] > table+.004) & (xyz[:, :, 2] < table+.38)
    pixels = np.argwhere(mask)
    points = xyz[mask]
    if len(points) < 200:
        return None
    # Bounded, deterministic fitting; only points in the current named object.
    remaining = points[::max(1, len(points)//2000)].copy()
    rng = np.random.default_rng(0)
    planes = []
    for _ in range(2):
        if len(remaining) < 80:
            return None
        best = np.zeros(len(remaining), bool)
        for _ in range(160):
            a, b, c = remaining[rng.choice(len(remaining), 3, replace=False)]
            normal = np.cross(b-a, c-a)
            length = np.linalg.norm(normal)
            if length < 1e-7:
                continue
            normal /= length
            inside = np.abs((remaining-a) @ normal) < .0015
            if inside.sum() > best.sum():
                best = inside
        if best.sum() < 80:
            return None
        face = remaining[best]
        centre = face.mean(0)
        _, _, axes = np.linalg.svd(face-centre, full_matrices=False)
        normal = axes[-1]
        if normal[2] < 0:
            normal = -normal
        local = (face-centre) @ axes[:2].T
        sizes = np.quantile(local, .98, axis=0)-np.quantile(local, .02, axis=0)
        if min(sizes) < .025 or max(sizes) > .45:
            return None
        residual = float(np.quantile(np.abs((face-centre) @ normal), .95))
        if residual > .002:
            return None
        planes.append(dict(point=centre.tolist(), normal=normal.tolist(),
                           pixels=int(best.sum()), size_m=sizes.tolist(), residual_m=residual))
        remaining = remaining[~best]
    n, m = [np.asarray(p['normal']) for p in planes]
    direction = np.cross(n, m)
    sine = np.linalg.norm(direction)
    if sine < .15:
        return None
    direction /= sine
    centres = np.array([p['point'] for p in planes])
    origin = np.linalg.solve(np.stack([n, m, direction]),
                             [n @ centres[0], m @ centres[1], direction @ centres.mean(0)])
    # A mathematical intersection far from observed surfaces is not evidence.
    # Only currently measured foreground pixels near it define the search crop.
    distance = np.linalg.norm(np.cross(points-origin, direction), axis=1)
    near = distance < .008
    if near.sum() < 20:
        return None
    along = (points[near]-origin) @ direction
    if np.ptp(along) < .035:
        return None
    selected = pixels[near][:, ::-1]
    lo, hi = selected.min(0), selected.max(0)
    # Padding is for current-image semantic identification, not geometry repair.
    lo = np.maximum(lo-10, 0)
    hi = np.minimum(hi+10, [w-1, h-1])
    return dict(source='current_two_plane_joint_search_only',
                bbox=(np.r_[lo, hi] / [w-1, h-1, w-1, h-1] * 1000).tolist(),
                hypothesis_origin=origin.tolist(), hypothesis_axis=direction.tolist(),
                observed_seam_neighbour_pixels=int(near.sum()), faces=planes,
                executable=False)


def measured_line_evidence(cloud, valid, line, table, robot_mask=None):
    """Necessary depth tests for model-selected endpoints, not semantic proof."""
    from robotwin_harness_v3 import SkillError
    xyz = np.asarray(cloud, float)
    mask = np.asarray(valid, bool).copy()
    endpoints = np.asarray(line, float)
    if (xyz.shape != mask.shape + (3,) or endpoints.shape != (2, 2)
            or not np.isfinite(endpoints).all() or (endpoints < 0).any() or (endpoints > 1000).any()):
        raise SkillError('Invalid current hinge calibration or endpoints')
    if robot_mask is not None:
        robot = np.asarray(robot_mask, bool)
        if robot.shape != mask.shape:
            raise SkillError('Hinge self-mask calibration mismatch')
        mask &= ~robot
    mask &= np.isfinite(xyz).all(2) & (xyz[:, :, 2] > table+.004)
    h, w = mask.shape
    uv = np.rint(endpoints * [w-1, h-1] / 1000).astype(int)
    if not mask[uv[:, 1], uv[:, 0]].all():
        raise SkillError('Hinge endpoints lack current nonrobot above-table depth')
    ends = xyz[uv[:, 1], uv[:, 0]]
    vector = ends[1]-ends[0]
    length = float(np.linalg.norm(vector))
    if not .02 < length < .5:
        raise SkillError('Implausible measured hinge span')
    axis = vector/length
    samples = np.unique(np.rint(np.linspace(uv[0], uv[1], 51)).astype(int), axis=0)
    x, y = samples.T
    distance = np.linalg.norm(np.cross(xyz[y, x]-ends[0], axis), axis=1)
    supported = mask[y, x] & (distance < .006)
    if supported.sum() < 8 or supported.mean() < .7:
        raise SkillError('Hinge line is not supported by a current continuous object surface')
    return dict(source='current_model_line_depth_admission',
                endpoints_world=ends.tolist(), axis=axis.tolist(), length_m=length,
                current_surface_fraction=float(supported.mean()), supported_pixels=int(supported.sum()))
