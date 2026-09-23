"""Current-frame disjoint bindings for a model-authored collection transfer.

Only distinct source names going to the same non-free destination/relation
are compared. Repeated use of one name, parts and inventory aliases are not
globally forced apart. The model must use one consistent name for a reused
physical object rather than counting that object twice under aliases.
"""
from robodawn.episodic_memory import object_key,free_region,same_visual_region


def collection_collisions(steps,step,bound,observation_id,candidate=None):
    if step.get('operation')!='transfer' or free_region(step.get('destination','')):
        return []
    source=object_key(step.get('source',''));destination=object_key(step.get('destination',''))
    binding=bound.get(source) if candidate is None else candidate
    if not binding or binding.get('observation_id')!=observation_id:return []
    peers=[]
    for other in steps:
        name=object_key(other.get('source',''))
        if (other.get('operation')!='transfer' or name==source or name in peers
                or object_key(other.get('destination',''))!=destination
                or other.get('relation')!=step.get('relation')):continue
        peers.append(name)
    collisions=[]
    for name in peers:
        other=bound.get(name)
        if other and other.get('observation_id')==observation_id and same_visual_region(binding,other):
            collisions.append(other)
    return collisions
