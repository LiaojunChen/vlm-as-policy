(() => {
  'use strict';
  const root=document.getElementById('workflow-app');if(!root)return;
  const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  // Display image basenames instead of machine-local paths; evidence downloads
  // retain the original bytes and hashes. Prompts and model outputs are unchanged.
  const pretty=v=>typeof v==='string'?v:JSON.stringify(v,(key,value)=>key==='image_paths'&&Array.isArray(value)?value.map(p=>p.split('/').pop()):value,2);
  const text=v=>typeof v==='object'&&v!==null?JSON.stringify(v):String(v??'');
  let index=[],flow=null,stepIndex=0,callIndex=0,camera='rgb',loadToken=0;
  const stepName=t=>t.action?.name||t.action?.skill||'动作';
  const outcomeLabel=f=>f.outcome==='success'?'成功样本':'失败样本';
  const meta=()=>index.find(x=>x.id===flow.id);
  async function loadIndex(){const r=await fetch('data/workflows/index.json');if(!r.ok)throw Error('index');return r.json();}
  const loaded=new Map();
  async function loadFlow(id){if(loaded.has(id))return loaded.get(id);const meta=index.find(x=>x.id===id);if(!meta)throw Error('unknown');const r=await fetch(meta.url);if(!r.ok)throw Error('workflow');const d=await r.json();loaded.set(id,d);return d;}
  function sensorTable(s){return `<div class="flow-sensors"><table><thead><tr><th>传感器/状态</th><th>当前值</th></tr></thead><tbody>${Object.entries(s||{}).map(([arm,v])=>`<tr><th>${esc(arm)} · 夹爪/接触</th><td><code>${esc(pretty(v))}</code></td></tr>`).join('')}</tbody></table></div>`;}
  function cameras(step){return `<div class="flow-camera-switch">${[['rgb','三路 RGB 原始观测'],['height','RGB-D 派生高度图']].map(([k,n])=>`<button data-camera="${k}" aria-pressed="${camera===k}">${n}</button>`).join('')}</div><div class="flow-cameras">${step.depth.map(d=>`<figure><a href="${esc(camera==='rgb'?step.rgb[d.camera]:d.preview)}" target="_blank" rel="noopener"><img src="${esc(camera==='rgb'?step.rgb[d.camera]:d.preview)}" alt="${d.camera} observation ${step.observation.observation_id} ${camera}"></a><figcaption>${d.camera} · ${camera==='rgb'?'RGB':'world_z − table_z'}<br><a href="${esc(d.raw_url)}" download>原始点云/遮罩 NPZ ↓</a></figcaption></figure>`).join('')}</div><p class="flow-disclosure">高度图为保存的 world_xyz 派生可视化，并非送入 VLM 的图像。蓝→青→黄→红对应高出桌面 0→0.4 m（超范围截断）；灰色为机器人自身遮罩，深色为无效点。NPZ 保留完整 world_xyz / valid / robot_self_mask，腕相机文件还含内外参。</p>`;}
  function callsFor(s){return flow.calls.filter(c=>c.step===s.step);}
  function callBody(c){if(!c)return '';
    const req=Object.fromEntries((c.request.messages||[]).map((m,i)=>[`${m.role}_${i}`,typeof m.content==='string'?m.content:m.content.filter(p=>p.type==='text').map(p=>p.text).join('\n')]));
    const out=c.raw_text||'(无 response.json；查看 error)';
    return `<div class="flow-call-meta"><span>call_${String(c.number).padStart(4,'0')}</span><span>${esc(c.kind)}</span><span>归属 source step ${c.step}</span>${c.response?.elapsed_s!=null?`<span>模型耗时 ${c.response.elapsed_s.toFixed(2)} s</span>`:''}</div><h5>真正送入 VLM 的 prompt 文本</h5><pre class="flow-prompt">${esc(Object.values(req).join('\n\n'))}</pre>${c.attachments.length?`<div class="flow-model-images">${c.attachments.map(a=>`<figure><img src="${esc(a.url)}" alt="model input ${esc(a.name)}"><figcaption>${esc(a.name)} · ${a.width}×${a.height} · 实际输入图片</figcaption></figure>`).join('')}</div>`:''}<h5>VLM 原始输出</h5><pre class="flow-raw">${esc(out)}</pre>${c.parsed_output?`<details><summary>解析后的 JSON 结构</summary><pre class="flow-json">${esc(pretty(c.parsed_output))}</pre></details>`:''}${c.rejection_feedback?`<div class="flow-alert"><strong>Harness 拒绝并触发重试：</strong> ${esc(c.rejection_feedback.reason)}<br><span>下一次调用：call_${String(c.rejection_feedback.next_call).padStart(4,'0')}；这个输出没有直接执行。</span></div>`:''}${c.error?`<div class="flow-alert"><strong>调用错误：</strong> ${esc(c.error.error)}</div>`:''}`;
  }
  function callView(c){
    const settings={model:c.request.model,temperature:c.request.temperature,max_tokens:c.request.max_tokens,response_format:c.request.response_format||null};
    return `<p class="flow-alert">${c.attachments.length?`本次请求实际附带 ${c.attachments.length} 张图片，见下方 JPEG；它可能是裁剪、遮罩或标注后的图像。`:'本次为纯文本请求，没有附带图片。请勿将上方三路 RGB 误认为本次模型输入。'}</p>`+callBody(c)+`<details><summary>请求参数、消息角色与 token 使用</summary><pre class="flow-json">${esc(pretty({settings,roles:c.request.messages.map(m=>m.role),usage:c.response?.response?.usage,matched_action_fields:c.matched_action_fields}))}</pre></details><div class="flow-evidence-links"><a href="${c.request_url}" target="_blank" rel="noopener">完整 request JSON ↗</a>${c.response_url?`<a href="${c.response_url}" target="_blank" rel="noopener">完整 response JSON ↗</a>`:''}</div>`;
  }
  function decorate(s,c){
    root.querySelectorAll('.flow-call-meta span').forEach(span=>{if(span.textContent.startsWith('模型耗时'))span.textContent=span.textContent.replace('模型耗时','调用墙钟耗时');});
    const summary=root.querySelector('.flow-summary');
    summary.innerHTML=`<strong>${flow.outcome==='success'?'✓ 最终完成':'× 最终未完成'}</strong><span>${esc(flow.result.instruction)}</span><a href="${meta().url}" target="_blank" rel="noopener">完整流程 JSON ↗</a><a href="${meta().evidence_url}" target="_blank" rel="noopener">原始文件 SHA-256 ↗</a>`;
    const cards=root.querySelectorAll('.flow-card');
    const pose=document.createElement('div');pose.className='flow-grid';
    pose.innerHTML=`<div><p>末端位姿（xyz / m + quaternion，保留原始顺序）与夹爪指令</p><pre class="flow-json">${esc(pretty(s.observation.endpose))}</pre></div><div><p>当前观察 / 桌面估计 / 官方状态</p><pre class="flow-json">${esc(pretty({observation_id:s.observation.observation_id,table_height_m:s.observation.table_height_m,success:s.observation.success}))}</pre><p class="flow-disclosure">上表为完整 harness 传感器记录。任务 planner 的 policy_sensor_view 会排除 opposed_contact 和 finger_normal_projection_range；具体送入模型的状态以实际 prompt 为准。</p></div>`;
    cards[0].querySelector('details').before(pose);
    const grounding=document.createElement('div');grounding.className='flow-alert';
    grounding.innerHTML=s.result.geometry?`<strong>RGB-D 几何结果：</strong> 从模型归一化像素位置和本步点云估计接触/支撑位置，再交给运动规划。完整数值见 geometry。<pre class="flow-json">${esc(pretty({tcp:s.result.geometry.tcp,bbox_px:s.result.geometry.bbox_px,top:s.result.geometry.top,bottom:s.result.geometry.bottom,camera:s.result.geometry.camera,candidates:s.result.geometry.candidates}))}</pre>`:'本步没有保存 geometry 字段；不要据此假设进行了新的视觉定位。move、rotate、home、present 等可沿用本体状态和既定动作参数。';
    cards[2].querySelector('h4').after(grounding);
    if(!callsFor(s).length){cards[1].insertAdjacentHTML('beforeend',`<p class="flow-alert">本步直接执行已有 mission 的阶段。${s.inherited_plan_call!==null?`最近的已接受任务计划是 call_${String(s.inherited_plan_call).padStart(4,'0')}。`:''}因此策略决策数不等于模型调用次数。</p>`);}
    const feedback=document.createElement('details');
    const next=flow.steps[stepIndex+1]?.observation||flow.result.final_observation;
    feedback.innerHTML=`<summary>${stepIndex===flow.steps.length-1?'终局真实观测与反馈':'下一步真正读到的 observation'}</summary><pre class="flow-json">${esc(pretty(next||'未保存'))}</pre>`;
    cards[3].append(feedback);
    if(stepIndex===flow.steps.length-1){
      const imgs=document.createElement('div');imgs.className='flow-cameras';
      imgs.innerHTML=Object.entries(flow.final_rgb).map(([name,url])=>`<figure><img loading="lazy" src="${url}" alt="final ${name}"><figcaption>最终 ${name}</figcaption></figure>`).join('');cards[3].append(imgs);
      const link=cards[3].querySelector('a[href="undefined"]');if(link)link.href=meta().url;
      if(flow.outcome==='failure')cards[3].insertAdjacentHTML('beforeend','<p class="flow-alert">本例已用完 30 次策略决策，官方 success 仍为 false。执行记录中没有 failure 并不表示扫描目标已完成；上游无效 JSON/技能重试也不一定出现在 trace.result.failure 中。</p>');
    }
    root.querySelectorAll('.flow-motion tbody tr').forEach((row,i)=>{const detail=document.createElement('details');detail.innerHTML=`<summary>目标与实际位姿</summary><pre class="flow-json">${esc(pretty(s.result.subactions[i]))}</pre>`;row.children[1].append(detail);});
    const timeline=root.querySelector('.flow-timeline'),current=timeline.querySelector('[aria-current=true]');
    if(innerWidth<=680)timeline.scrollLeft=stepIndex*136;else timeline.scrollTop=Math.max(0,current.offsetTop-timeline.offsetTop-100);
    const share=document.createElement('button');share.className='text-button';share.textContent='复制此步骤链接';share.dataset.flowShare='';summary.append(share);
  }
  function actionCard(s){const a=s.action,r=s.result;const subs=(r.subactions||[]).map((x,i)=>`<tr><td>${i+1}</td><td>${esc(x.name)}</td><td>${x.motion_ok===true?'成功':x.success===true?'完成':x.planner_status?esc(pretty(x.planner_status)):'—'}</td><td>${x.position_error_m!=null?`${Number(x.position_error_m).toFixed(4)} m`:''}</td></tr>`).join('');return `<div class="flow-card"><h4><small>HARNESS / EXECUTE</small>${esc(stepName(s))} · ${esc(a.arm||'—')} ${a.target?`· ${esc(a.target)}`:''}</h4><div class="flow-grid"><div><p>Harness 收到的动作对象（含 grounding、观测编号、任务阶段）</p><pre class="flow-json">${esc(pretty(a))}</pre></div><div><p>执行结果与环境反馈</p><pre class="flow-json">${esc(pretty(r))}</pre></div></div>${subs?`<details open><summary>运动执行器返回的 ${r.subactions.length} 个子步骤</summary><div class="flow-motion"><table><thead><tr><th>#</th><th>子步骤</th><th>规划/完成</th><th>误差</th></tr></thead><tbody>${subs}</tbody></table></div></details>`:''}<div class="flow-checks"><span>skill_success：${r.skill_success===true?'true':r.skill_success===false?'false':'—'}</span><span>official success：${r.success===true?'true':r.success===false?'false':'—'}</span>${r.failure?`<span class="flow-alert">failure：${esc(r.failure)}</span>`:'<span>failure：无</span>'}</div></div>`;}
  function renderBase(){if(!flow)return;const s=flow.steps[stepIndex],cs=callsFor(s);callIndex=Math.min(callIndex,Math.max(0,cs.length-1));const c=cs[callIndex];
    root.innerHTML=`<div class="flow-choices">${index.map(m=>`<button class="flow-choice" data-flow="${m.id}" aria-pressed="${m.id===flow.id}"><span>${m.outcome==='success'?'✓':'×'} ${outcomeLabel(m)}</span><strong>${esc(m.task)}</strong><small>seed ${m.seed} · ${m.decisions} 次决策 · ${m.calls} 次模型调用</small></button>`).join('')}</div><div class="flow-route"><span>RGB-D / 本体状态 / 接触</span><i>→</i><span>VLM prompt + 图像</span><i>→</i><span>原始 JSON 输出</span><i>→</i><span>Harness 校验 / grounding</span><i>→</i><span>技能执行器</span><i>→</i><span>环境 success</span></div><div class="flow-summary"><strong>${flow.outcome==='success'?'✓ 完成':'× 未完成'}</strong><span>${esc(flow.result.instruction)}</span><a href="${esc(flow.result.attempt_evidence?`../${flow.result.attempt_evidence}`:'#')}">原始证据路径 ↗</a><span>调用归属：${esc(flow.association.note)}</span></div><div class="flow-shell"><aside class="flow-sidebar"><h3>See → Think → Act → Observe</h3><div class="flow-budget"><span style="width:${(stepIndex+1)/flow.steps.length*100}%"></span></div><p class="micro">step ${stepIndex+1} / ${flow.steps.length}</p><div class="flow-timeline">${flow.steps.map((x,i)=>`<button class="flow-step ${i===flow.steps.length-1?'ending':''}" data-step="${i}" aria-current="${i===stepIndex}"><span>OBS ${x.observation.observation_id} · 决策 ${i+1}</span><b>${esc(stepName(x))}</b><small>${x.result.skill_success===true?'技能成功':x.result.failure?'有反馈':'执行'}</small></button>`).join('')}</div></aside><div class="flow-stage-area"><div class="flow-step-head"><h3>Step ${s.step} · ${esc(stepName(s))}</h3><div class="flow-controls"><button data-prev ${stepIndex===0?'disabled':''}>←</button><button data-next ${stepIndex===flow.steps.length-1?'disabled':''}>→</button></div></div><div class="flow-card"><h4><small>OBSERVE / SENSOR INPUT</small>Harness 在本步看到什么</h4><p>这次观察来自环境的三路 RGB、末端位姿、夹爪关节与接触状态；harness 同时从 RGB-D 计算点云、桌面高度和 robot self-mask。所有内容来自该步保存的 observation ${s.observation.observation_id}。</p>${cameras(s)}${sensorTable(s.observation.sensors)}<details><summary>完整 observation JSON</summary><pre class="flow-json">${esc(pretty(s.observation))}</pre></details></div><div class="flow-card"><h4><small>THINK / VLM CALLS</small>本步发生的 ${cs.length} 次模型调用</h4><div class="flow-call-list">${cs.map((x,i)=>`<button data-call="${i}" aria-pressed="${i===callIndex}">call_${String(x.number).padStart(4,'0')} · ${esc(x.kind)}</button>`).join('')||'<span class="micro">本步没有模型调用；使用了上一轮计划或 harness 内部执行。</span>'}</div>${c?callView(c):''}<p class="flow-disclosure">请求中的图片文件名、prompt 文本和响应原文均来自本轮 calls 目录。若出现 rejected 输出，只有修复后的 action 才进入 Harness execute。</p></div>${actionCard(s)}<div class="flow-card"><h4 class="${flow.outcome==='failure'&&stepIndex===flow.steps.length-1?'flow-end-title failure':'flow-end-title'}"><small>OUTCOME / OFFICIAL EVALUATOR</small>${stepIndex===flow.steps.length-1?(flow.outcome==='success'?'任务完成':'任务未完成'):'下一步继续'}</h4>${stepIndex===flow.steps.length-1?`<p>终止原因：<strong>${esc(flow.result.end_reason)}</strong></p><p>${flow.result.error?`Harness/模型错误：<code>${esc(flow.result.error)}</code>`:'环境官方 success 最终为 '+(flow.result.success?'true':'false')+'。'}</p><video class="flow-final-video" controls preload="metadata" src="media/official500/${esc(flow.result.task)}/repeat_${String(flow.result.repeat_index||0).padStart(2,'0')}/video.mp4"></video><div class="flow-evidence-links"><a href="${esc(flow.raw['result.json'])}" target="_blank">result.json ↗</a><a href="${esc(flow.raw['trace.jsonl'])}" target="_blank">trace.jsonl ↗</a><a href="${esc(flow.result.trace_url)}" target="_blank">本页结构化 workflow ↗</a></div>`:'<p>Harness 记录了本步的结果，并把下一次 observe 反馈给 planner；左侧时间线可以继续前进。</p>'}</div></div></div><p class="flow-disclosure">证据边界：原始日志没有原生 step→call_id 和视频时间戳；本页按 request 文件与 observation frame 的保存时间归属调用，并用 action.raw / grounding_raw 的 JSON 语义交叉核对。这个限制在页面中公开，避免把近似时间线说成精确同步。</p>`;
  }
  function render(){renderBase();if(flow)decorate(flow.steps[stepIndex]);}
  function updateHash(){const call=callsFor(flow.steps[stepIndex])[callIndex];history.replaceState(null,'',`#workflow=${flow.id}&step=${stepIndex}${call?`&call=${call.number}`:''}`);}
  async function fromHash(scroll){
    const params=new URLSearchParams(location.hash.slice(1)),id=params.get('workflow');
    if(!index.some(x=>x.id===id))return false;
    const token=++loadToken,next=await loadFlow(id);if(token!==loadToken)return true;
    flow=next;stepIndex=Math.max(0,Math.min(flow.steps.length-1,Number(params.get('step'))||0));
    callIndex=Math.max(0,callsFor(flow.steps[stepIndex]).findIndex(c=>c.number===Number(params.get('call'))));render();
    if(scroll)document.getElementById('workflow').scrollIntoView({block:'start'});return true;
  }
  root.addEventListener('click',async e=>{
    const flowBtn=e.target.closest('[data-flow]');
    if(flowBtn){const token=++loadToken;try{const next=await loadFlow(flowBtn.dataset.flow);if(token!==loadToken)return;flow=next;stepIndex=0;callIndex=0;render();updateHash();}catch(err){const status=document.createElement('p');status.className='flow-error';status.textContent='加载失败，请重新点击样本重试。';root.prepend(status);}return;}
    if(e.target.closest('[data-flow-share]')){updateHash();try{await navigator.clipboard.writeText(location.href);e.target.textContent='链接已复制';}catch(_){e.target.textContent='请复制地址栏链接';}return;}
    const sb=e.target.closest('[data-step]');if(sb){stepIndex=Number(sb.dataset.step);callIndex=0;render();updateHash();return;}
    const cb=e.target.closest('[data-call]');if(cb){callIndex=Number(cb.dataset.call);render();updateHash();return;}
    if(e.target.closest('[data-prev]')){stepIndex=Math.max(0,stepIndex-1);callIndex=0;render();updateHash();}
    if(e.target.closest('[data-next]')){stepIndex=Math.min(flow.steps.length-1,stepIndex+1);callIndex=0;render();updateHash();}
    const cam=e.target.closest('[data-camera]');if(cam){camera=cam.dataset.camera;render();}
  });
  window.addEventListener('hashchange',()=>{if(location.hash.startsWith('#workflow='))fromHash(true).catch(()=>{});});
  loadIndex().then(async x=>{index=x;if(!await fromHash(true)){flow=await loadFlow(index[0].id);render();}}).catch(e=>{root.innerHTML=`<div class="flow-error">完整流程数据加载失败：${esc(e.message)}。可从结果矩阵打开原始录像。</div>`;});
})();
