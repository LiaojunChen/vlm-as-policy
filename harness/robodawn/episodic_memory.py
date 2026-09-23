"""Episode-local memory of policy observations and execution feedback only.

No simulator, asset metadata, task registry, evaluator or cross-episode data.
Historical boxes are identity hints, NEVER executable current-frame geometry.
"""
from copy import deepcopy
import json
import re


def object_key(description):
    return ' '.join(str(description).lower().split())


def free_region(description):
    return bool(re.search(r'\b(table|tabletop|workspace)\b', description, re.I))


def same_visual_region(first, second):
    """Conservative duplicate detection; containment alone is not identity."""
    if first.get('camera','head_camera')!=second.get('camera','head_camera'):return False
    a, b = first['bbox'], second['bbox']
    intersection = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    area_a = max(0, a[2]-a[0])*max(0, a[3]-a[1])
    area_b = max(0, b[2]-b[0])*max(0, b[3]-b[1])
    return intersection / max(area_a+area_b-intersection, 1e-9) > .65


class EpisodicMemory:
    def __init__(self):
        self.objects = {}
        self.events = []
        from robodawn.held_search import HeldSearchMemory
        self.held_search = HeldSearchMemory()
        self.latest_observation = None
        self.completion_review = None
        from robodawn.intent_memory import IntentMemory
        self.intentions = IntentMemory()

    def remember_visual(self, action, observation_id):
        # Generic endpoint names ("heel", "tip") are not global identities.
        # Directed geometry already keeps them scoped to the source object.
        if action.get('grounding_role')=='orientation_endpoint':
            return
        target = action.get('target')
        if not target or free_region(target):
            return
        key = object_key(target)
        entry = self.objects.setdefault(key, {'description': target, 'state': 'observed'})
        # A rim, heel or tool tip is not the whole object's identity box.
        component = (action.get('grasp_part', 'body') != 'body'
                     or action.get('skill') in ('place', 'press', 'arc')
                     or action.get('grounding_role') == 'orientation_endpoint')
        if not component or 'last_visual' not in entry:
            entry['last_visual'] = dict(bbox=list(action['bbox']), point=list(action['point']),
                                        observation_id=observation_id, component_only=component,
                                        camera=action.get('camera','head_camera'))
        entry['last_seen_observation'] = observation_id
        for field in ('verified_appearance','identity_verification_observation_id','preferred_grasp_part','inventory_identity'):
            if field in action:entry[field]=deepcopy(action[field])

    def preferred_grasp_part(self, target):
        return self.objects.get(object_key(target),{}).get('preferred_grasp_part')

    def current_binding(self, target, observation_id, camera=None):
        """Reuse identity candidates only when they came from this exact frame."""
        entry=self.objects.get(object_key(target),{});visual=entry.get('last_visual',{})
        if entry.get('held_by') or visual.get('component_only',True) or visual.get('observation_id')!=observation_id:return None
        if camera is not None and visual.get('camera','head_camera')!=camera:return None
        result=dict(skill='reach',arm='auto',target=target,bbox=deepcopy(visual['bbox']),point=deepcopy(visual['point']),
                    grounding_role='entity_identity',grounding_source='same_observation_model_memory',observation_id=observation_id,
                    camera=visual.get('camera','head_camera'))
        for field in ('preferred_grasp_part','inventory_identity','verified_appearance','identity_verification_observation_id'):
            if field in entry:result[field]=deepcopy(entry[field])
        return result

    def search_region(self, target, observation_id=None, camera='head_camera'):
        """An uncertain region for CURRENT-image search, never action coordinates."""
        entry=self.objects.get(object_key(target),{})
        visual=entry.get('last_visual',{})
        region=entry.get('last_release_region')
        if entry.get('held_by'):return None
        if not region:
            if (observation_id is None or entry.get('state')!='observed' or visual.get('component_only',True)
                    or visual.get('observation_id',observation_id)>=observation_id):return None
            region=visual  # Untouched stationary object: uncertain CURRENT-frame search only.
        released=region.get('after_observation_id')
        if (released is not None and not visual.get('component_only',True)
                and visual.get('observation_id',-1)>released):region=visual
        if region.get('camera','head_camera')!=camera:return None
        # A wrist moves with the arm. Even unchanged objects move in its image;
        # historical wrist boxes cannot define a current search crop.
        if camera!='head_camera':return None
        box=region.get('bbox')
        if not isinstance(box,(list,tuple)) or len(box)!=4:return None
        if not (0<=box[0]<box[2]<=1000 and 0<=box[1]<box[3]<=1000):return None
        return deepcopy(dict(bbox=box,status='uncertain_memory_guided_current_image_search'))

    def feedback(self, action, result):
        self.held_search.feedback(self.latest_observation,action,result)
        self.intentions.feedback(action, result)
        skill, target, arm = action.get('skill'), action.get('target'), action.get('arm')
        event = {k: deepcopy(action[k]) for k in ('skill', 'target', 'arm', 'grasp_part', 'approach', 'relation') if k in action}
        event.update(skill_success=result.get('skill_success'), failure=result.get('failure'))
        # Retain intended relation as intention, not a proven scene relation.
        intent = action.get('mission_intent', {})
        if skill == 'place' and intent.get('source'):
            event['moved_source'] = intent['source']
            if result.get('skill_success') and action.get('release',True):
                entry=self.objects.get(object_key(intent['source']))
                region=(result.get('geometry') or {}).get('pending_release_region')
                if entry is not None:entry.pop('last_release_region',None)
                if region is None and action.get('relation','on') in ('on','inside') and 'bbox' in action and 'point' in action:
                    region={k:deepcopy(action[k]) for k in ('bbox','point')}
                if entry is not None and region is not None:
                    entry['last_release_region']=deepcopy(region)
                    entry['last_release_region']['status']='expected_from_release_not_yet_visually_verified'
                    entry['last_release_region']['after_observation_id']=action.get('observation_id')
                    entry['last_release_region']['camera']=action.get('camera','head_camera')
        if self.events and {k: v for k, v in self.events[-1].items() if k != 'repeats'} == event:
            self.events[-1]['repeats'] = self.events[-1].get('repeats', 1)+1
        else:
            self.events.append(event)
        self.events = self.events[-12:]
        # Contact is the only authority for holding. Failed motions may still
        # have moved an object, so never retain a confident stationary box.
        if target and skill in ('pick', 'grasp_handle', 'push', 'press', 'place', 'arc') and result.get('subactions'):
            entry = self.objects.get(object_key(target))
            if entry:
                entry['state'] = 'may_have_moved_reobserve'
        for side, state in result.get('sensors', {}).items():
            for entry in self.objects.values():
                if entry.get('held_by') == side and not state.get('holding'):
                    entry.pop('held_by', None)
                    entry['state'] = 'released_or_lost_reobserve'
            name = state.get('remembered_object')
            if state.get('holding') and name:
                entry = self.objects.setdefault(object_key(name), {'description': name})
                entry.update(state='held_contact_verified', held_by=side)

    def context(self, target=None):
        entries = list(self.objects.values())
        if target is not None:
            key = object_key(target)
            entries = [v for k, v in self.objects.items() if k == key or v.get('held_by')]
        context = {'objects': entries[-12:], 'recent_outcomes': self.events[-6:] if target is None else
                   [e for e in self.events if object_key(e.get('target', '')) == object_key(target)][-3:]}
        if target is None and self.intentions.entries:
            context['original_execution_intentions'] = self.intentions.context()
        if target is None and self.completion_review is not None:
            context['completion_stage_review'] = self.completion_review
        return deepcopy(context)

    def prompt(self, target=None, include_coordinates=True):
        context = self.context(target)
        if not include_coordinates:
            # A current-image crop has a different coordinate system. Feeding
            # old full-image boxes alongside it causes copied coordinates and
            # recursively shrinking search windows. Keep identity/history only.
            for entry in context['objects']:
                entry.pop('last_visual', None)
                entry.pop('last_release_region', None)
        if not context['objects'] and not context['recent_outcomes'] and not context.get('original_execution_intentions') and not context.get('completion_stage_review'):
            return ''
        review_note=''
        if context.get('completion_stage_review'):
            from robodawn.completion_review import REVIEW_NOTE
            review_note=REVIEW_NOTE
        return ('\nEPISODIC MEMORY (only past images and executed feedback; not ground truth): '
                + json.dumps(context, separators=(',', ':'))
                + '\nUse memory to preserve object identity and avoid repeated failures. '
                + ('Historical boxes refer to their tagged full camera image, not a crop or another camera. ' if include_coordinates else
                   'Historical coordinates are intentionally omitted; return coordinates ONLY in the currently shown image. ')
                + 'Objects may have moved: re-localize in the CURRENT image. A successful skill or attempted placement '
                  'does not prove the goal or spatial relation. A last_release_region is a newer, uncertain search hint, not a confirmed object box. '
                  'Positional words in an object identity describe its INITIAL position, which may change after moving it. '
                  'The held source is NOT its destination.'
                + (' Original execution intentions survive replanning: include every not_executed_successfully source operation. '
                   'skill_executed_goal_unverified records only execution feedback, never goal completion.'
                   if context.get('original_execution_intentions') else '')+review_note)
