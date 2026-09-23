"""Export two complete, audited sensor → model → harness → feedback workflows.

Calls have no native action IDs. Recover their grouping from original observation/
request file mtimes (one-second resolution) and cross-check final raw/grounding
outputs semantically. Never claim those mtimes are precise execution timestamps.
"""
import bisect
import hashlib
import json
import re
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
HARNESS=ROOT.parent/'phyRSI/robodawn_robotwin_harness_v72'
COHORT='repeat500_official_sampling_maxsteps30_20260922'
RUN=HARNESS/'results'/COHORT
OUT=ROOT/'site/data/workflows'
SAMPLES=[('stack_blocks_three',0,'success'),('scan_object',0,'failure')]

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,ensure_ascii=False,separators=(',',':')))
def parsed(s):
    try:return json.loads(re.sub(r'^```(?:json)?\s*|\s*```$','',s.strip()))
    except (ValueError,TypeError):return None
def prompts(request):
    return '\n'.join(m['content'] if isinstance(m['content'],str) else '\n'.join(p.get('text','') for p in m['content']) for m in request['messages'])
def role(text):
    if 'Observe this current image and list' in text:return '场景物体清单'
    if text.startswith('Make an ordered manipulation plan'):return '任务计划 / 重规划'
    if 'The TOP panel is the full current scene' in text:return '物体身份配对核验'
    if 'Locate the ' in text:return '视觉定位 / 接触点'
    if 'Control the two Aloha arms using ONE next skill' in text:return '失败恢复 / 下一技能'
    return '模型调用'

def build():
    index=[]
    for task,repeat,outcome in SAMPLES:
        eid=f'{task}__{repeat:02d}';folder=RUN/'robodawn'/task/f'repeat_{repeat:02d}/attempts/attempt_000'
        dest=OUT/eid;dest.mkdir(parents=True,exist_ok=True)
        evidence=[]
        def copy(src,relative):
            target=dest/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,target)
            evidence.append(dict(source=str(src.relative_to(HARNESS)),published=str(target.relative_to(ROOT/'site')),sha256=sha(src)))
            return str(target.relative_to(ROOT/'site'))
        result=read(folder/'result.json');rows=[json.loads(l) for l in (folder/'trace.jsonl').read_text().splitlines() if l.strip()]
        assert len(rows)==result['policy_decisions'] and bool(result['success'])==(outcome=='success')
        raw={n:copy(folder/n,'raw/'+n) for n in ['result.json','trace.jsonl']}
        frame_times=[(folder/'frames'/Path(row['observation']['image_paths'][0]).name).stat().st_mtime_ns for row in rows]
        assert frame_times==sorted(frame_times) and len(set(frame_times))==len(frame_times)
        calls=[]
        for f in sorted((folder/'calls').glob('call_*.request.json')):
            request=read(f);response_file=f.with_name(f.name.replace('.request.json','.response.json'))
            response=read(response_file) if response_file.exists() else None
            error_file=f.with_name(f.name.replace('.request.json','.error.json'))
            number=int(f.name[5:9]);step=bisect.bisect_right(frame_times,f.stat().st_mtime_ns)-1
            assert 0<=step<len(rows)
            attachments=[]
            for m in request['messages']:
                if isinstance(m['content'],list):
                    for part in m['content']:
                        if part.get('type')=='image_url':
                            name=part['image_url']['url'];assert Path(name).name==name
                            p=folder/'calls'/name;url=copy(p,'calls/'+name)
                            with Image.open(p) as im:width,height=im.size
                            attachments.append(dict(name=name,url=url,width=width,height=height))
            raw_text=response['response']['choices'][0]['message']['content'] if response else None
            output=parsed(raw_text) if raw_text else None
            action=rows[step]['action'];matches=[]
            for key in ['raw','grounding_raw']:
                if action.get(key) and output is not None and parsed(action[key])==output:matches.append(key)
            c=dict(number=number,step=step,kind=role(prompts(request)),request=request,response=response,
                   request_url=copy(f,'calls/'+f.name),response_url=copy(response_file,'calls/'+response_file.name) if response else None,
                   attachments=attachments,raw_text=raw_text,parsed_output=output,matched_action_fields=matches,
                   request_mtime_ns=f.stat().st_mtime_ns,rejection_feedback=None)
            if error_file.exists():c['error']=read(error_file);c['error_url']=copy(error_file,'calls/'+error_file.name)
            calls.append(c)
        assert len(calls)==result['model_calls'] and [c['number'] for c in calls]==list(range(len(calls)))
        for c,nxt in zip(calls,calls[1:]):
            text=prompts(nxt['request'])
            # Read the actual validation reason appended to the following prompt.
            if c['step']==nxt['step'] and c['raw_text'] and c['raw_text'].strip() in text:
                m=re.search(r'\n(?:REASON|Reason): ([\s\S]*?)(?:\nChoose a corrected action|\nCorrect the JSON plan|\nReturn the corrected|$)',text)
                if m:c['rejection_feedback']=dict(next_call=nxt['number'],reason=m.group(1))
        steps=[];latest_plan=None
        for row in rows:
            i=row['step'];cs=[c for c in calls if c['step']==i]
            for c in cs:
                if c['kind']=='任务计划 / 重规划' and not c['rejection_feedback']:latest_plan=c['number']
            # Raw text may be normalized by the structured decoder; use JSON semantic equality.
            if row['action'].get('raw') or row['action'].get('grounding_raw'):
                assert any(c['matched_action_fields'] for c in cs), f'Unverified call/action grouping {eid} step {i}'
            obs=row['observation'];rgb={Path(p).name.split('_',1)[1].removesuffix('.png'):f"media/official500/{task}/repeat_{repeat:02d}/frame/{Path(p).with_suffix('.webp').name}" for p in obs['image_paths']}
            depth=[]
            for camera in ['head_camera','left_camera','right_camera']:
                stem=f"{obs['observation_id']:04d}"+('' if camera=='head_camera' else '_'+camera)
                source=folder/'frames'/f'{stem}_rgbd.npz'
                with np.load(source) as npz:
                    xyz=npz['world_xyz'];valid=npz['valid'];own=npz['robot_self_mask'];z=xyz[:,:,2]-obs['table_height_m']
                    t=np.nan_to_num(np.clip(z/0.4,0,1));palette=np.array([[38,57,90],[57,150,159],[234,196,92],[207,83,48]],dtype=float)
                    scaled=t*3;low=np.clip(scaled.astype(int),0,2);frac=(scaled-low)[...,None]
                    color=(palette[low]*(1-frac)+palette[low+1]*frac).astype(np.uint8)
                    color[~valid]=[27,30,33];color[own]=[170,170,170]
                    preview=dest/'depth'/f'{stem}_height.webp';preview.parent.mkdir(exist_ok=True)
                    Image.fromarray(color).save(preview,'WEBP',lossless=True)
                    depth.append(dict(camera=camera,preview=str(preview.relative_to(ROOT/'site')),
                                      raw_url=copy(source,'depth/'+source.name),shape=list(xyz.shape),valid_pixels=int(valid.sum()),robot_pixels=int(own.sum()),
                                      keys={key:list(npz[key].shape) for key in npz.files},
                                      xyz_min=xyz[valid].min(axis=0).tolist(),xyz_max=xyz[valid].max(axis=0).tolist(),
                                      table_height_m=obs['table_height_m']))
            steps.append(dict(**row,calls=[c['number'] for c in cs],rgb=rgb,depth=depth,
                              observation_mtime_ns=frame_times[i],inherited_plan_call=latest_plan))
        final=result.get('final_observation',{})
        data=dict(id=eid,outcome=outcome,result=result,steps=steps,calls=calls,raw=raw,
                  final_rgb={Path(p).name.split('_',1)[1].removesuffix('.png'):f"media/official500/{task}/repeat_{repeat:02d}/frame/{Path(p).with_suffix('.webp').name}" for p in final.get('image_paths',[])},
                  association=dict(method='request_file_mtime_between_observation_frame_mtimes',precision='one_second',
                                   native_call_step_id=False,semantic_cross_checks=sum(bool(c['matched_action_fields']) for c in calls),
                                   note='调用归属由原始文件保存时间恢复，并核对 action.raw / grounding_raw 的 JSON 语义；日志没有原生 step→call_id 或视频同步时间戳。'),
                  sensor_note='三路 RGB、末端位姿、夹爪关节位置、仿真接触状态由观测记录保存；RGB-D 点云/机器人自身遮罩供 harness 几何运算。VLM 每次实际收到的文本与图片以 request 为准；初始 mission 计划请求可能仅含文本。')
        write(dest/'workflow.json',data);write(dest/'evidence.json',evidence)
        index.append(dict(id=eid,task=task,outcome=outcome,seed=result['seed'],decisions=len(steps),calls=len(calls),
                          url=str((dest/'workflow.json').relative_to(ROOT/'site')),evidence_url=str((dest/'evidence.json').relative_to(ROOT/'site'))))
    write(OUT/'index.json',index)
    print(json.dumps(index,ensure_ascii=False))

if __name__=='__main__':build()
