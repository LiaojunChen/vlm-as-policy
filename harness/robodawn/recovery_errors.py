"""Conservative classification of nested planning/grounding failures.

Wrapper labels do not establish a physical observation failure. A mixed
multi-camera error containing a generation/transport fault abstains from
camera motion; no state, coordinates, retry budget or task identity is inferred.
"""


def failure_kind(error):
    text=str(error).lower()
    if any(word in text for word in ('http ', 'connection', 'timed out', 'timeout',
                                     'service unavailable', 'network', 'out of memory')):
        return 'service'
    if any(word in text for word in ('json', 'schema', 'generation', 'decode', 'token budget')):
        return 'generation_or_format'
    if any(word in text for word in ('no reachable', 'unreachable', 'tracking error', 'ik_fail')):
        return 'motion'
    if any(word in text for word in ('visible', 'occlud', 'identity', 'identities',
                                     'visual object', 'no depth', 'no above-table',
                                     'empty grasp', 'without measured object contact',
                                     'contact point is on the robot', 'nonrobot above-table depth')):
        return 'perception'
    return 'unknown'
