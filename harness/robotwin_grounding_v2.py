"""RGB-guided, depth-grounded RoboTwin skills for the v2 evaluation.

The policy selects a target in RGB. Camera position/depth and robot kinematics are
sensor observations; object actor poses and expert actions are never read here.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from robotwin_bridge import PolicyOutputError


SIDES = ('left', 'right')
MOVE = {
    'MV_LEFT': (-1, 0, 0), 'MV_RIGHT': (1, 0, 0),
    'MV_FWD': (0, 1, 0), 'MV_BACK': (0, -1, 0),
    'MV_UP': (0, 0, 1), 'MV_DOWN': (0, 0, -1),
}
# Local tool +X points down; the orientation can be reached by both Aloha arms.
DOWN_QUAT = [math.sqrt(.5), 0., math.sqrt(.5), 0.]
FRAME_CONTEXT_V2 = (
    'HEAD RGB is 320x240; LEFT and RIGHT images are wrist cameras. '
    'World +X points right in HEAD, +Y points toward the back of the table '
    '(toward image top), +Z points up. MV_FWD=+Y and MV_BACK=-Y. '
    'The robot pose is measured in metres. If a motion reports failed planning '
    'or less than 5 mm travel, select a different reachable target; do not '
    'repeat the same failed command.\n'
)


def _cell(text: str, prefix: str, max_col: int, max_row: int) -> tuple[int, int]:
    match = re.fullmatch(rf'{prefix}\s*(\d{{1,2}})\s*R\s*(\d{{1,2}})', text.strip(), re.I)
    if not match:
        raise PolicyOutputError(f'Missing {prefix}<column>R<row> cell code: {text[:200]!r}')
    col, row = int(match.group(1)), int(match.group(2))
    if not (0 <= col <= max_col and 0 <= row <= max_row):
        raise PolicyOutputError(f'Out-of-range cell code: {text[:200]!r}')
    return col, row


def _grid(image: Image.Image, columns: int, rows: int) -> Image.Image:
    image = image.convert('RGB').copy()
    draw = ImageDraw.Draw(image, 'RGBA')
    w, h = image.size
    for col in range(1, columns):
        x = round(col * w / columns)
        draw.line((x, 0, x, h - 1), fill=(210, 15, 15, 95), width=1)
    for row in range(1, rows):
        y = round(row * h / rows)
        draw.line((0, y, w - 1, y), fill=(210, 15, 15, 95), width=1)
    for col in range(columns):
        draw.text((int((col + .2) * w / columns), 1), str(col),
                  fill=(0, 0, 0, 255), stroke_width=1,
                  stroke_fill=(255, 255, 255, 255))
    for row in range(rows):
        draw.text((1, int((row + .3) * h / rows)), str(row),
                  fill=(0, 0, 0, 255), stroke_width=1,
                  stroke_fill=(255, 255, 255, 255))
    return image


def _color_name(description: str) -> str | None:
    words = re.findall(r'\b(?:yellow|red|green|blue|orange|purple|black)\b',
                       description.lower())
    # In "yellow button of blue bell", the first colour names the contact part.
    return words[0] if words else None


def _refine_color(image: np.ndarray, centre: tuple[int, int], colour: str,
                  radius: int = 35) -> tuple[int, int] | None:
    u, v = centre
    h, w = image.shape[:2]
    x0, x1 = max(0, u - radius), min(w, u + radius + 1)
    y0, y1 = max(0, v - radius), min(h, v + radius + 1)
    crop = image[y0:y1, x0:x1]
    if not crop.size:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    ranges = {
        'yellow': (19, 39), 'orange': (8, 20), 'green': (35, 88),
        'blue': (90, 135), 'purple': (130, 168),
    }
    if colour == 'red':
        mask = (((hue <= 9) | (hue >= 170)) & (sat >= 95) & (val >= 65))
    elif colour == 'black':
        mask = (val <= 75)
    elif colour in ranges:
        lo, hi = ranges[colour]
        mask = ((hue >= lo) & (hue <= hi) & (sat >= 70) & (val >= 60))
    else:
        return None
    if mask.sum() < 5:
        return None
    count, labels, stats, centres = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), 8)
    candidates = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < 5:
            continue
        cu, cv = centres[label]
        pixel = (int(round(cu + x0)), int(round(cv + y0)))
        distance = math.dist(pixel, centre)
        candidates.append((distance - min(area, 300) / 100.0, pixel))
    return min(candidates)[1] if candidates else None


class GridGrounder:
    def __init__(self, client, directory: Path):
        self.client = client
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.index = 0

    def locate(self, image_path: str, description: str) -> dict:
        image = Image.open(image_path).convert('RGB')
        if image.size != (320, 240):
            raise ValueError(f'Unexpected HEAD resolution {image.size}')
        index = self.index
        self.index += 1
        coarse_image = _grid(image, 16, 12)
        coarse_path = self.directory / f'ground_{index:04d}_coarse.png'
        coarse_image.save(coarse_path)
        prompt = (
            f'Find the visible part described as: {description}. In the HEAD image, '
            'identify its physical contact point, not a shadow or background. '
            'The red grid has columns 0-15 and rows 0-11 labelled in the image. '
            'Reply with exactly one cell code C<column>R<row>, e.g. C10R7. '
            'No explanation.'
        )
        coarse_raw = self.client.complete_text(prompt, np.asarray(coarse_image),
                                               max_tokens=64, temperature=0).raw_text
        col, row = _cell(coarse_raw, 'C', 15, 11)
        cu, cv = col * 20 + 10, row * 20 + 10
        crop = image.crop((cu - 30, cv - 30, cu + 30, cv + 30))
        fine_image = _grid(crop.resize((240, 240)), 3, 3)
        fine_path = self.directory / f'ground_{index:04d}_fine.png'
        fine_image.save(fine_path)
        fine_prompt = (
            f'This is a magnified 60x60 crop around {description}. '
            'Find that same physical contact point. The red grid has columns '
            'and rows 0-2 labelled in the image. Reply with exactly one cell '
            'code F<column>R<row>, e.g. F1R0. No explanation.'
        )
        fine_raw = self.client.complete_text(fine_prompt, np.asarray(fine_image),
                                             max_tokens=64, temperature=0).raw_text
        fine_col, fine_row = _cell(fine_raw, 'F', 2, 2)
        pixel = (cu + (fine_col - 1) * 20, cv + (fine_row - 1) * 20)
        colour = _color_name(description)
        refined = (_refine_color(np.asarray(image), pixel, colour)
                   if colour else None)
        if refined:
            pixel = refined
        if not (0 <= pixel[0] < 320 and 0 <= pixel[1] < 240):
            raise PolicyOutputError(f'Grounded pixel outside image: {pixel}')
        result = {'pixel': list(pixel), 'coarse_cell': [col, row],
                  'fine_cell': [fine_col, fine_row], 'coarse_raw': coarse_raw,
                  'fine_raw': fine_raw, 'colour_refinement': colour if refined else None,
                  'description': description,
                  'coarse_image': str(coarse_path), 'fine_image': str(fine_path)}
        (self.directory / f'ground_{index:04d}.json').write_text(
            json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        return result


class GroundedBridge:
    def __init__(self, env, instruction: str, directory: Path):
        self.env, self.instruction, self.directory = env, instruction, Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.index = 0
        self.last_observation = None

    def observe(self) -> dict:
        obs = self.env.get_obs()
        self.last_observation = obs
        paths = []
        for camera in ('head_camera', 'left_camera', 'right_camera'):
            path = self.directory / f'{self.index:04d}_{camera}.png'
            Image.fromarray(obs['observation'][camera]['rgb']).save(path)
            paths.append(str(path))
        return {'instruction': self.instruction, 'image_paths': paths,
                'endpose': obs['endpose'], 'step': self.index,
                'success': bool(self.env.eval_success or self.env.check_success())}

    def pixel_world(self, pixel: tuple[int, int]) -> list[float]:
        cam = self.env.cameras.static_camera_list[self.env.cameras.head_camera_id]
        u, v = pixel
        picture = cam.get_picture('Position')
        point = np.asarray(picture[v, u, :3], dtype=float)
        if not np.isfinite(point).all() or point[2] >= -.10:
            raise PolicyOutputError(f'No valid depth at pixel {pixel}')
        world = cam.get_model_matrix() @ np.array([*point, 1.0])
        if not np.isfinite(world[:3]).all():
            raise PolicyOutputError(f'Nonfinite world point at pixel {pixel}')
        return world[:3].tolist()

    def _take(self, side: str, pose: list[float], gripper: float,
              name: str) -> dict:
        before = self.env.get_obs()['endpose']
        statuses = {}
        originals = {s: getattr(self.env.robot, s + '_plan_path') for s in SIDES}
        for arm in SIDES:
            original = originals[arm]
            def capture(target, *args, original=original, arm=arm, **kwargs):
                value = original(target, *args, **kwargs)
                statuses[arm] = value.get('status')
                return value
            setattr(self.env.robot, arm + '_plan_path', capture)
        target = []
        for arm in SIDES:
            target.extend((pose if arm == side else before[arm + '_endpose']) +
                          [gripper if arm == side else before[arm + '_gripper']])
        try:
            self.env.take_action(target, action_type='ee')
        finally:
            for arm in SIDES:
                setattr(self.env.robot, arm + '_plan_path', originals[arm])
        after = self.env.get_obs()['endpose']
        actual = math.dist(before[side + '_endpose'][:3],
                           after[side + '_endpose'][:3])
        commanded = math.dist(before[side + '_endpose'][:3], pose[:3])
        success = bool(self.env.eval_success or self.env.check_success())
        return {'name': name, 'target_ee': target,
                'planner_status': statuses, 'actual_translation_m': actual,
                'commanded_translation_m': commanded, 'endpose_after': after,
                'motion_ok': (statuses.get(side) == 'Success' and
                              (commanded < .005 or actual >= min(.005, .25 * commanded)))
                             or success,
                'success': success}

    def _pose(self, point: list[float], height: float) -> list[float]:
        return [float(point[0]), float(point[1]), float(point[2] + height),
                *DOWN_QUAT]

    def execute(self, action: dict) -> dict:
        side = action.get('arm')
        if side not in SIDES:
            raise PolicyOutputError(f'Invalid arm {side!r}')
        name = action.get('name')
        if name not in ('press_at', 'pick_at', 'place_at', 'reach_at', 'move_token'):
            raise PolicyOutputError(f'Invalid grounded skill {name!r}')
        steps = []
        grounding = action.get('grounding')
        point = self.pixel_world(tuple(action['pixel'])) if name != 'move_token' else None
        def take(label, pose, grip):
            row = self._take(side, pose, grip, label)
            steps.append(row)
            return row
        before = self.env.get_obs()['endpose']
        current = before[side + '_endpose']
        gripper = before[side + '_gripper']
        if name == 'move_token':
            token = action.get('token')
            if token not in MOVE:
                raise PolicyOutputError(f'Invalid movement token {token!r}')
            for size in (.04, .02, .01):
                target = current.copy()
                target[:3] = (np.asarray(current[:3]) +
                              np.asarray(MOVE[token]) * size).tolist()
                # Preflight is read-only: select the largest reachable step.
                planner = getattr(self.env.robot, side + '_plan_path')
                if planner(target).get('status') == 'Success':
                    take(f'{token}_{size:.2f}m', target, gripper)
                    break
            else:
                return {'ok': False, 'success': False, 'subactions': [],
                        'failure': 'No reachable step at 4, 2 or 1 cm',
                        'token': token}
        elif name == 'press_at':
            take('close', current, 0.)
            if not steps[-1]['success']:
                row = take('hover', self._pose(point, .20), 0.)
                if row['motion_ok'] and not row['success']:
                    take('press', self._pose(point, .12), 0.)
        elif name == 'reach_at':
            take('reach', self._pose(point, .20), gripper)
        elif name == 'pick_at':
            take('open', current, 1.)
            row = take('hover', self._pose(point, .20), 1.)
            if row['motion_ok'] and not row['success']:
                contact_height = max(.08, .74 + .12 + .02 - point[2])
                row = take('descend', self._pose(point, contact_height), 1.)
                if row['motion_ok'] and not row['success']:
                    pose = self.env.get_obs()['endpose'][side + '_endpose']
                    take('close', pose, 0.)
                    if not steps[-1]['success']:
                        pose = self.env.get_obs()['endpose'][side + '_endpose'].copy()
                        pose[2] += .10
                        take('lift', pose, 0.)
        elif name == 'place_at':
            row = take('hover', self._pose(point, .22), gripper)
            if row['motion_ok'] and not row['success']:
                row = take('descend', self._pose(point, .17), gripper)
                if row['motion_ok'] and not row['success']:
                    pose = self.env.get_obs()['endpose'][side + '_endpose']
                    take('open', pose, 1.)
        self.index += 1
        success = bool(self.env.eval_success or self.env.check_success())
        return {'ok': all(s['motion_ok'] for s in steps), 'success': success,
                'subactions': steps, 'grounding': grounding,
                'world_point': point, 'sim_actions': len(steps),
                'failure': next((s['name'] + ': ' + str(s['planner_status'])
                                 for s in steps if not s['motion_ok']), None)}
