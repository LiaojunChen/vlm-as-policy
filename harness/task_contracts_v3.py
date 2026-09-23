"""Public task completion semantics, independent of sampled object state/expert actions.

These hints describe goals and robot capabilities. They do not contain object poses,
contact-point IDs, asset identities, expert trajectories or per-seed answers.
"""
COMMON='Use verified tactile state and fresh images. Complete every named object. For placement, release and retreat to inspect the final arrangement. For lifting, presenting or scanning, retain the required objects; do not release them merely to end the task.'
CONTRACTS={
 'move_playingcard_away':'Move the card box outward to a side edge of the table, then release. A small central displacement is insufficient; both grippers must finish open.',
 'move_stapler_pad':'The stapler must lie flat, precisely centred on the coloured pad, with its long edge parallel to the pad long edge. Both grippers finish open.',
 'place_mouse_pad':'The mouse must lie flat, precisely centred on the coloured pad, with its long edge parallel to the pad long edge. Both grippers finish open.',
 'pick_dual_bottles':'Both bottles must be retained upright and raised near the front centre. Pick and present the first, keep holding it, then pick and present the second. Do not re-grasp with an occupied arm.',
 'pick_diverse_bottles':'Both bottles must be retained upright and raised near the front centre. Pick and present the first, keep holding it, then pick and present the second. Do not re-grasp with an occupied arm.',
 'adjust_bottle':'Raise the bottle while keeping it towards the specified arm side of the workspace; retain it in the gripper.',
 'stack_blocks_two':'Put the red block on the table first, then place the green block exactly on top. Align their horizontal centres; release and leave both grippers open.',
 'stack_blocks_three':'Build a single vertical tower in the specified bottom-to-top colour order. Place each block centred on the preceding block and release.',
 'stack_bowls_two':'Nest the designated upper bowl upright into the lower bowl. Grasp a rim, not the empty centre; release both grippers at completion.',
 'stack_bowls_three':'Nest all bowls upright into one stack. Grasp a rim, not the empty centre; release both grippers at completion.',
 'place_empty_cup':'Use PUSH to slide the cup along the table onto the coaster centre while keeping it upright; the destination is the coaster. Finish with both grippers open. Hovering above the coaster is insufficient.',
 'place_object_basket':'First put the object into the upright basket, then lift the basket slightly while retaining the object inside. Grasp the basket handle or rim.',
 'place_can_basket':'Put the can inside the basket, then lift the upright basket by its handle; keep the can inside.',
 'place_bread_skillet':'Place the bread inside the skillet and lift the skillet by its handle while keeping it level.',
 'place_container_plate':'Keep the container upright and centre its base on the plate, then release.',
 'place_bread_basket':'Put every requested bread into the basket interior, then release both arms.',
 'place_cans_plasticbox':'Put both requested cans into the box interior without knocking the first one out. Finish with both grippers open.',
 'put_bottles_dustbin':'Put every requested bottle into the bin interior. Keep track of which objects have already been transferred.',
 'place_dual_shoes':'Put both shoes into the shoebox with their toes pointing left, then release.',
 'rotate_qrcode':'Finish with the sign resting on the table, its QR face towards the robot, and both grippers open. Holding it in the air is not completion.',
 'open_laptop':'Grasp the lid edge without lifting the whole laptop. Open it by following the lid hinge arc about a horizontal axis.',
 'open_microwave':'Grasp the door handle without lifting the appliance, then pull the handle along the door hinge arc.',
 'put_object_cabinet':'Grasp the drawer handle without lifting the cabinet; pull horizontally to open the drawer. Release its handle, then put the requested object inside.',
 'lift_pot':'Use both handles to lift the pot upright. A single central grasp is not sufficient.',
 'grab_roller':'Use one arm on each end of the roller and lift together, keeping both grippers closed.',
 'handover_block':'Use the first hand to present the block to the other hand. The receiving hand must acquire contact before the first releases; then place on the pad.',
 'handover_mic':'Hold the microphone body for the receiving hand, acquire contact with the receiving hand before releasing the first.',
 'turn_switch':'Contact the movable switch lever and move it through its travel; tapping a fixed housing is insufficient.',
 'beat_block_hammer':'Grasp the hammer handle, then contact the block with the hammer head.',
 'stamp_seal':'Grasp the seal handle and press its stamping face squarely onto the named pad.',
 'scan_object':'Hold the requested object and scanner in separate hands; aim the scanner toward the object while bringing them close.',
 'hanging_mug':'Grasp the mug body, orient its handle opening toward the rack peg, guide the peg through the handle and release only when supported.',
}

def contract(task):return COMMON+' '+CONTRACTS.get(task,'')
