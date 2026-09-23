"""Task-agnostic goal-oriented interface for the unchanged language model."""
PROMPT = '''Plan the requested FINAL arrangement and arm occupancy from the task and image.
Return only JSON {"steps":[...]}, with a short, nonredundant ordered plan.
Use consistent object descriptions that distinguish similar objects by colour or initial position.
Resolve which object moves and which object supports it before planning. They must be DIFFERENT physical objects.
Choose operations from this API (all keys shown are top-level fields within a step):
- transfer: operation, source, destination, relation, arm, grasp_part. Picks source, places at destination, RELEASES and homes automatically. Do not add a lift, release or home for this same transfer.
- stack: operation, sources (BOTTOM to TOP order), arms (one per source), grasp_part. Leaves the bottom where it is and moves each higher object onto the preceding one. Do not also transfer the same objects. Use a separate transfer only if the bottom needs a specified new location.
- arrange: operation, sources (desired LEFT to RIGHT order), arms, grasp_part. Moves each object to a separate table slot. Do not also transfer these same objects.
- lift: operation, source, arm, location (stay/centre/side), grasp_part. Pick and KEEP holding, only when holding/presenting is actually required, not as preparation for transfer.
- handover: operation, source, arm (DONOR left/right), receiver (OTHER left/right), grasp_part. Picks with the donor if needed, presents, then the receiver grasps a currently visible free part of the SAME object. Releases the donor ONLY after both hands have verified contact on that object. Do not use transfer-to-hand, two independent lifts, or bimanual_lift for passing. If placement follows, transfer with the receiver.
- bimanual_lift: operation, source, left_source, right_source, grasp_part. BOTH hands lift ONE object using its two distinct handles/ends. Two independent objects instead require two lifts.
- press: operation, source, arm. EMPTY-hand button/lever contact, includes approach. Never pick a button.
- slide: operation, source, destination, relation, arm. EMPTY-hand pushing.
- tool_contact: operation, source, contact_part, destination, relation, arm, grasp_part. Picks the tool and uses its working head/tip WITHOUT releasing it. For hammering/stamping/striking, do not substitute transfer or add a preliminary lift. Name contact_part precisely, distinct from the handle.
- action: operation, action (object with skill and parameters). Only for actions not covered above:
  grasp_handle (arm,target,grasp_part,approach) holds an articulated door/lid/drawer without lifting;
  move (arm,delta [dx,dy,dz], length <=0.2m); rotate (arm,axis x/y/z,angle signed <=90 degrees);
  arc (arm,target=hinge,axis,angle signed <=90 degrees); shake (arm,axis,amplitude 0.02..0.06,cycles 1..4);
  present (arm,location centre/side); open/close/home (arm). Moving an object requires first holding it; home requires an EMPTY arm.
arm: left/right/auto. Obey explicit arm requirements; otherwise auto.
grasp_part: body for a solid compact object, rim for a bowl, handle for a basket/pan/tool, edge for a thin edge.
relation: on (top), inside (opening), left_of/right_of/in_front_of/behind (beside, not on top).
Free-table destination: a visible free table region, relation=on.
For explicit tip/toe/front direction, transfer also needs orientation={"from_part":description of rear end,"to_part":description of front end,"facing":"left"/"right"/"front"/"back"}; both ends belong to the source.
Complete every requested object and stage without duplicating an achieved relation. Keep arms free for later stages.
When recovering, use current image, contact state and memory to plan only remaining corrections. No guessed hidden state or coordinates.
'''
