"""Retain an original action prefix; enable measured release at a declared row.

Diagnostic-only comparison, not the autonomous policy or evaluator entrypoint.
No task identity selects behavior: the caller declares the intervention row.
"""
import argparse
import sys

import measured_release
from robotwin_harness_v3 import Bridge
import probe_retained_prefix


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--enable-from-row',type=int,required=True)
    args,remaining=parser.parse_known_args()
    if args.enable_from_row<0:raise ValueError('Nonnegative intervention row required')
    execute=Bridge.execute;snapshot=measured_release.measured_release_pose
    index={'row':0}
    def release(endpose,side,planned):
        if index['row']<args.enable_from_row:
            return list(planned),dict(mode='original_planned_pose_release_retained_prefix',
                                      diagnostic_intervention_row=args.enable_from_row)
        pose,evidence=snapshot(endpose,side,planned)
        evidence['diagnostic_intervention_row']=args.enable_from_row
        return pose,evidence
    def step(bridge,action):
        result=execute(bridge,action);index['row']+=1
        return result
    saved_argv=sys.argv
    measured_release.measured_release_pose=release;Bridge.execute=step
    try:
        sys.argv=[saved_argv[0],*remaining]
        probe_retained_prefix.main()
    finally:
        sys.argv=saved_argv
        measured_release.measured_release_pose=snapshot;Bridge.execute=execute


if __name__=='__main__':main()
