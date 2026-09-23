"""Frame-safe routing over the three existing, unchanged RGB-D cameras."""
from pathlib import Path
from robotwin_harness_v3 import SkillError

CAMERAS = ('head_camera', 'left_camera', 'right_camera')


def camera_of(value):
    camera = value.get('camera', 'head_camera')
    if camera not in CAMERAS:
        raise SkillError('Unknown observation camera: '+str(camera))
    return camera


def in_camera(obs, camera):
    if camera not in CAMERAS:
        raise SkillError('Unknown observation camera: '+str(camera))
    paths = obs['image_paths']
    by_name = obs.get('camera_paths') or dict(zip(CAMERAS, paths))
    if camera not in by_name:
        raise SkillError('Requested camera is absent from the current observation')
    return dict(obs, camera=camera, camera_paths=by_name,
                image_paths=[by_name[camera]]+[p for p in paths if p != by_name[camera]])


def evidence_path(image_path, observation_id, camera):
    path = Path(image_path)
    if camera not in CAMERAS or path.name != f'{observation_id:04d}_{camera}.png':
        return None
    suffix = 'rgbd' if camera == 'head_camera' else camera+'_rgbd'
    return path.with_name(f'{observation_id:04d}_{suffix}.npz')


def candidate_views(obs, action):
    if 'camera' in action:
        return [camera_of(action)]
    # Wrist views are used only following a successful active observation.
    # No change to the established head-only path otherwise.
    active = obs.get('inspection_cameras', [])
    return list(dict.fromkeys([c for c in active if c in CAMERAS]+['head_camera']))


def observed_world_side(obs, action):
    import numpy as np
    camera=camera_of(action);view=in_camera(obs,camera)
    path=evidence_path(view['image_paths'][0],obs['observation_id'],camera)
    if path is None or not path.is_file():raise SkillError('No current depth for wrist-view arm assignment')
    with np.load(path,allow_pickle=False) as data:
        height,width=data['valid'].shape
        x,y=np.rint(np.asarray(action['point'])*[width-1,height-1]/1000).astype(int)
        if not data['valid'][y,x]:raise SkillError('No depth for wrist-view arm assignment')
        point=data['world_xyz'][y,x]
        if not np.isfinite(point).all():raise SkillError('Invalid current wrist-view depth')
    return 'left' if point[0]<0 else 'right'
