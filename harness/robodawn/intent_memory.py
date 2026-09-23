"""Durable execution intentions, not a scene-state or success oracle.

The first accepted model plan provides identities and operations. Replanning
may change order, arms and support topology, but must not silently forget an
object whose required operation has never produced successful feedback.
"""
from copy import deepcopy
from robodawn.episodic_memory import object_key
from robotwin_harness_v3 import SkillError


COMPATIBLE = {
    'lift': {'lift', 'transfer', 'tool_contact', 'bimanual_lift', 'handover'},
    'transfer': {'transfer', 'slide'},
    'slide': {'transfer', 'slide'},
    'press': {'press'},
    'tool_contact': {'tool_contact'},
    'bimanual_lift': {'bimanual_lift'},
    'handover': {'handover'},
}


class IntentMemory:
    def __init__(self):
        self.entries = []
        self.initialized = False

    def initialize(self, steps):
        if self.initialized:
            return
        seen = set()
        for step in steps:
            operation = step.get('operation')
            source = step.get('source')
            if operation not in COMPATIBLE or not source:
                continue
            key = (operation, object_key(source))
            if key in seen:
                continue
            seen.add(key)
            entry = {k: deepcopy(step[k]) for k in
                     ('operation', 'source', 'destination', 'relation', 'orientation', 'receiver') if k in step}
            entry['execution_status'] = 'not_executed_successfully'
            self.entries.append(entry)
        self.initialized = True

    def pending(self):
        return [entry for entry in self.entries
                if entry['execution_status'] == 'not_executed_successfully']

    def validate_replan(self, steps, sensors):
        missing = []
        for entry in self.pending():
            source = object_key(entry['source'])
            operation = entry['operation']
            if operation == 'lift' and any(
                    state.get('holding') and object_key(state.get('remembered_object', '')) == source
                    for state in sensors.values()):
                continue
            if not any(object_key(step.get('source', '')) == source
                       and step.get('operation') in COMPATIBLE[operation] for step in steps):
                missing.append({'source': entry['source'], 'operation': operation})
        if missing:
            raise SkillError('Replan omitted unexecuted original intentions: '+str(missing)+
                             '. Keep these exact object identities in remaining SOURCE operations. '
                             'Failed attempts and naming an object only as a destination do not execute its operation. '
                             'Order, feasible arms and support order may change within the task instruction.')

    def feedback(self, action, result):
        if not result.get('skill_success'):
            return
        intent = action.get('mission_intent', {})
        source = object_key(intent.get('source', action.get('target', '')))
        skill = action.get('skill')
        for entry in self.pending():
            if object_key(entry['source']) != source:
                continue
            operation = entry['operation']
            executed = False
            if operation == 'lift':
                executed = skill == 'pick' and any(
                    state.get('holding') and object_key(state.get('remembered_object', '')) == source
                    for state in result.get('sensors', {}).values())
            elif operation in ('transfer', 'slide'):
                executed = ((skill == 'place' and action.get('release', True)) or skill == 'push')
                executed = executed and not intent.get('regrasp_stage') and not action.get('regrasp_stage')
            elif operation == 'press':
                executed = skill == 'press'
            elif operation == 'tool_contact':
                executed = skill == 'place' and action.get('use_contact_part') and not action.get('release', True)
            elif operation == 'bimanual_lift':
                executed = skill == 'dual_move' and intent.get('operation') == 'bimanual_lift'
            elif operation == 'handover':
                executed = (skill == 'handover' and
                    (result.get('geometry') or {}).get('handover',{}).get('phase')=='receiver_retained_after_release')
            if executed:
                entry['execution_status'] = 'skill_executed_goal_unverified'

    def context(self):
        return deepcopy(self.entries)
