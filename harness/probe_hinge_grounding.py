"""Same-frame hinge-line diagnosis. No simulator actions or official score."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from benchmark_clients import AuditedClient
from robodawn.camera_views import in_camera, evidence_path
from robotwin_harness_v3 import parse_json
from source_layout import ROOT, evaluation_sources


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episode', type=Path, required=True)
    parser.add_argument('--rows', type=int, nargs='+', default=[0, 1])
    parser.add_argument('--cameras', nargs='+', default=['head_camera'])
    parser.add_argument('--target', required=True)
    parser.add_argument('--parent-crop', action='store_true')
    parser.add_argument('--joint-search', action='store_true', help='Use current two-plane seam only as a model search hint')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError('Keep diagnostics inside this version')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'code_hashes.json').write_text(json.dumps({
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in evaluation_sources()}, indent=2))
    rows = [json.loads(line) for line in (args.episode / 'trace.jsonl').read_text().splitlines()]
    client = AuditedClient('zdtaichu', output / 'calls')
    report = dict(diagnostic_only=True, official_task_run=False,
                  source_episode=str(args.episode.resolve()), queries=[])
    for index in args.rows:
        for camera in args.cameras:
            obs = in_camera(rows[index]['observation'], camera)
            image = np.asarray(Image.open(obs['image_paths'][0]))
            prompt = ('Locate the visible rotation hinge joining the moving part and stationary base of '
                      + args.target + '. Use only this current image. Identify the actual pivot seam, '
                      'NOT the outer free edge of the moving part, table, or robot. '
                      'Give TWO distinct visible points along the SAME straight hinge axis, on solid '
                      'object pixels near its opposite ends. Never guess through occlusion. '
                      'Coordinates are normalized 0..1000 along image width and height. '
                      'Return only JSON {"visible":true,"hinge_line":[[x1,y1],[x2,y2]]} '
                      'or {"visible":false} if two hinge points cannot be seen.')
            item = dict(row=index, observation_id=obs['observation_id'], camera=camera)
            try:
                crop = None
                if args.parent_crop or args.joint_search:
                    from robodawn.semantic_planner import ground
                    parent = ground(client, obs, dict(skill='reach', arm='left', target=args.target,
                                                     grounding_role='entity_identity', camera=camera))
                    item['parent_grounding'] = parent
                    height, width = image.shape[:2]
                    search_box = parent['bbox']
                    if args.joint_search:
                        from articulation_geometry import planar_joint_search
                        path = evidence_path(obs['image_paths'][0], obs['observation_id'], camera)
                        with np.load(path, allow_pickle=False) as data:
                            valid = data['valid'].astype(bool)
                            if 'robot_self_mask' in data:
                                valid &= ~data['robot_self_mask'].astype(bool)
                            hint = planar_joint_search(data['world_xyz'], valid, search_box, obs['table_height_m'])
                        item['joint_search'] = hint
                        if hint is not None:
                            search_box = hint['bbox']
                            prompt += (' Current depth suggests two nonparallel faces may meet near this crop. '
                                       'This is ONLY a search hypothesis, not a known hinge; verify the actual '
                                       'joining seam visually and reject it if absent. Do not select a free rim.')
                    box = np.asarray(search_box) * [width-1, height-1, width-1, height-1] / 1000
                    lo = np.maximum(np.floor(box[:2]-12), 0).astype(int)
                    hi = np.minimum(np.ceil(box[2:]+12), [width-1, height-1]).astype(int)
                    crop = lo, hi, width, height
                    image = image[lo[1]:hi[1]+1, lo[0]:hi[0]+1]
                    item['crop_pixels'] = [*lo.tolist(), *hi.tolist()]
                    prompt += ' This is a crop of the CURRENT object; returned coordinates refer to this crop.'
                raw = client.complete_text(prompt, image, max_tokens=180, temperature=0).raw_text
                item['raw'] = raw
                located = parse_json(raw)
                item['located'] = located
                if located.get('visible') is not False:
                    line = np.asarray(located['hinge_line'], float)
                    if line.shape != (2, 2) or not np.isfinite(line).all() or (line < 0).any() or (line > 1000).any():
                        raise ValueError('Malformed normalized hinge line')
                    height, width = image.shape[:2]
                    if crop is not None:
                        lo, hi, width, height = crop
                        line = (line * (hi-lo) / 1000 + lo) / [width-1, height-1] * 1000
                        item['full_frame_hinge_line'] = line.tolist()
                    pixels = np.rint(line * [width-1, height-1] / 1000).astype(int)
                    path = evidence_path(obs['image_paths'][0], obs['observation_id'], camera)
                    with np.load(path, allow_pickle=False) as data:
                        cloud = data['world_xyz']
                        valid = data['valid'].astype(bool)
                        robot = data.get('robot_self_mask', np.zeros_like(valid)).astype(bool)
                    xyz = cloud[pixels[:, 1], pixels[:, 0]]
                    vector = xyz[1]-xyz[0]
                    length = float(np.linalg.norm(vector))
                    samples = np.unique(np.rint(np.linspace(pixels[0], pixels[1], 31)).astype(int), axis=0)
                    sx, sy = samples.T
                    sample_cloud = cloud[sy, sx]
                    item.update(pixels=pixels.tolist(), endpoints_world=xyz.tolist(), length_m=length,
                                axis=(vector/max(length, 1e-12)).tolist(),
                                endpoint_valid=valid[pixels[:, 1], pixels[:, 0]].tolist(),
                                endpoint_robot=robot[pixels[:, 1], pixels[:, 0]].tolist(),
                                endpoint_height_above_table=(xyz[:, 2]-obs['table_height_m']).tolist(),
                                line_samples=len(samples), line_valid_fraction=float(valid[sy, sx].mean()),
                                line_robot_fraction=float(robot[sy, sx].mean()),
                                line_above_table_fraction=float((sample_cloud[:, 2] > obs['table_height_m']+.004).mean()))
                    from articulation_geometry import measured_line_evidence
                    try:
                        item['depth_admission'] = measured_line_evidence(cloud, valid, line, obs['table_height_m'], robot)
                    except Exception as exc:
                        item['depth_rejection'] = str(exc)
                    if item.get('joint_search'):
                        hint = item['joint_search']
                        residual = np.linalg.norm(np.cross(xyz-np.asarray(hint['hypothesis_origin']),
                                                          np.asarray(hint['hypothesis_axis'])), axis=1)
                        item['endpoint_to_plane_intersection_m'] = residual.tolist()
            except Exception as exc:
                item['error'] = f'{type(exc).__name__}: {exc}'
            report['queries'].append(item)
            (output / 'report.json').write_text(json.dumps(report, indent=2))
            print(json.dumps(item), flush=True)


if __name__ == '__main__':
    main()
