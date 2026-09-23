import unittest
import numpy as np
from robotwin_harness_v3 import validate,SkillError,grasp_quat,ee_from_tcp,tcp_from_ee,Bridge
class V3Tests(unittest.TestCase):
 def test_tcp_calibration_roundtrip_for_rotations(self):
  for yaw in [0,.5,1.57,3.14]:
   for pitch in [0,.5,1.57]:
    tcp=[.1,-.1,.8];q=grasp_quat(yaw,pitch)
    np.testing.assert_allclose(tcp_from_ee(ee_from_tcp(tcp,q)),tcp,atol=1e-10)
 def test_negative_pixels_and_missing_identity(self):
  for a in [dict(skill='pick',arm='right',bbox=[-1,0,500,500],point=[250,250],target='cube'),
            dict(skill='pick',arm='right',bbox=[0,0,500,500],point=[600,250],target='cube'),
            dict(skill='pick',arm='right',bbox=[0,0,500,500],point=[250,250])]:
   with self.assertRaises(SkillError):validate(a)
 def test_motion_bounds(self):
  for a in [dict(skill='rotate',arm='right',axis='y',angle=100),dict(skill='move',arm='right',delta=[.2,.2,0]),dict(skill='move',arm='right',delta=[float('nan'),0,0]),dict(skill='pull',arm='right')]:
   with self.assertRaises(SkillError):validate(a)
 def test_geometry_rejects_tabletop_as_object(self):
  b=Bridge.__new__(Bridge);b.depth_id=1;b.table=.74;b.landmarks={};b.cloud=np.zeros((240,320,3));b.cloud[:,:,2]=.74;b.valid=np.ones((240,320),bool)
  a=dict(skill='pick',arm='right',bbox=[0,0,1000,1000],point=[500,500],target='cube',observation_id=1)
  with self.assertRaisesRegex(SkillError,'No above-table'):b.geometry(a)
  a['observation_id']=0
  with self.assertRaisesRegex(SkillError,'Stale'):b.geometry(a)
if __name__=='__main__':unittest.main()

class GroundingRegression(unittest.TestCase):
 def test_placement_keeps_holding_arm_when_destination_on_other_side(self):
  import json,tempfile
  from pathlib import Path
  from types import SimpleNamespace
  from PIL import Image
  from robotwin_harness_v3 import RobodawnPlanner
  responses=[dict(skill='place',arm='left',target='pad',bbox=[600,600,800,800],point=[700,700]),
             dict(arm='right',target='DESTINATION',bbox=[600,600,800,800],point=[700,700],approach='top',support='table')]
  class Client:
   def complete_text(self,*args,**kwargs):return SimpleNamespace(raw_text=json.dumps(responses.pop(0)))
  with tempfile.TemporaryDirectory() as temp:
   path=Path(temp)/'rgb.png';Image.new('RGB',(320,240)).save(path)
   obs=dict(success=False,image_paths=[str(path)]*3,observation_id=7,endpose={},sensors={'left':{'holding':True}},table_height_m=.74)
   action=RobodawnPlanner(Client(),'place object on pad').plan(obs,[])
   self.assertEqual(action['arm'],'left');self.assertEqual(action['support'],'table');self.assertEqual(action['target'],'pad')

 def test_tactile_requires_two_fingers_on_same_dynamic_body(self):
  from types import SimpleNamespace as S
  from robotwin_harness_v3 import Bridge
  class PhysxRigidStaticComponent:
   def __init__(self,name,i):self.entity=S(name=name,per_scene_id=i)
  class PhysxRigidDynamicComponent(PhysxRigidStaticComponent):pass
  joint=lambda n:S(name=n,child_link=S(name=n))
  joints=[joint('fl_finger1'),joint('fl_finger2'),joint('fr_finger1'),joint('fr_finger2')]
  entity=S(get_active_joints=lambda:joints,get_qpos=lambda:[.02]*4)
  robot=S(left_gripper=[(j,1,0) for j in joints[:2]],right_gripper=[(j,1,0) for j in joints[2:]],left_entity=entity,right_entity=entity)
  contacts=[];b=Bridge.__new__(Bridge);b.held={'left':None,'right':None};b.env=S(robot=robot,scene=S(get_contacts=lambda:contacts))
  def touch(finger,body):return S(bodies=[PhysxRigidDynamicComponent(finger,99),body],points=[S(impulse=[.01,0,0])])
  table=PhysxRigidStaticComponent('table',1);obj=PhysxRigidDynamicComponent('anonymous',2)
  contacts[:]=[touch('fl_finger1',table),touch('fl_finger2',table)]
  self.assertFalse(b.sensors()['left']['holding'])
  contacts[:]=[touch('fl_finger1',obj)];self.assertFalse(b.sensors()['left']['holding'])
  contacts.append(touch('fl_finger2',obj));self.assertTrue(b.sensors()['left']['holding'])
