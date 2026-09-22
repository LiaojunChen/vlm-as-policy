/* All metrics come from build_data.py; text content is escaped before rendering. */
(() => {
  'use strict';
  const D = window.REPORT_DATA, A = D.aggregate;
  const $ = id => document.getElementById(id);
  const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const pct = (n,d) => d ? `${(n/d*100).toFixed(0)}%` : '—';
  const reason = {environment_success:'官方判定成功',decision_budget:'决策预算耗尽',invalid_model_action:'动作无效终止',wall_time_budget:'墙钟预算耗尽'};
  const media = (task,file) => `media/${encodeURIComponent(task)}/${file === "video" ? "video.mp4" : file}`;
  let selected, currentStep = 0;
  const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
  function goReplay() { window.scrollTo({top:$('replay').getBoundingClientRect().top + window.scrollY - 100, behavior:reduced?'instant':'smooth'}); }
  function selectTask(task, scroll=false) { const e=D.episodes.find(e=>e.task===task); if(!e)return; renderEpisode(e); history.replaceState(null,'',`#episode=${encodeURIComponent(task)}`);if(scroll)goReplay(); }
  const initial = D.episodes.find(e=>e.task==='stack_blocks_three');
  $('hero-image').src=media(initial.task,`frame/${initial.final_frames.head_camera}`);
  const metrics=[['72%','官方任务成功率',`${A.successes} / ${A.total} · restored 完整批次`],['50 / 50','任务已完成','36 成功 · 14 未成功 · 0 运行错误'],[A.mean_decisions.toFixed(1),'平均策略决策数','每个 episode 上限 30 次'],[`${A.mean_elapsed_s.toFixed(1)} s`,'平均墙钟耗时',`中位数 ${A.median_elapsed_s.toFixed(1)} s`]];
  $('metrics').innerHTML=metrics.map(([n,l,s])=>`<div class="metric"><strong>${n}</strong><span>${l}</span><small>${s}</small></div>`).join('');
  $('task-matrix').innerHTML=D.episodes.map((e,i)=>`<button class="task-cell ${e.success?'':'failed'}" data-task="${e.task}" title="${esc(e.task)} · ${e.success?'成功':'未成功'}" aria-label="${esc(e.task)}，${e.success?'成功':'未成功'}，查看回放">${String(i+1).padStart(2,'0')}</button>`).join('');
  $('family-bars').innerHTML=D.groups.map(g=>`<div class="family-row"><div class="bar-label"><span>${g.name}</span><b>${g.successes} / ${g.total} · ${pct(g.successes,g.total)}</b></div><div class="bar-track"><span style="width:${g.successes/g.total*100}%"></span></div></div>`).join('');
  const best=[...D.groups].sort((a,b)=>b.successes/b.total-a.successes/a.total)[0];
  $('outcome-insight').innerHTML=`<strong>${best.name} ${best.successes}/${best.total} 成功。</strong> 这组固定场景中，该任务族完成率最高；其余任务的失败集中于预算耗尽和动作无效终止。分组样本量有限，需结合逐任务证据解读。`;
  function table(){
    const term=$('search').value.trim().toLowerCase(), status=$('status-filter').value, sort=$('sort').value;
    const rows=D.episodes.filter(e=>(status==='all'||e.success===(status==='success'))&&`${e.task} ${e.instruction} ${e.family}`.toLowerCase().includes(term));
    rows.sort((a,b)=>sort==='task'?a.task.localeCompare(b.task):b[sort]-a[sort]);
    $('row-count').textContent=`显示 ${rows.length} / ${D.episodes.length} 个任务；当前结果中成功 ${rows.filter(e=>e.success).length} 个。表格内滚动可浏览全部记录。`;
    $('task-rows').innerHTML=rows.length?rows.map(e=>`<tr><td class="task-name">${esc(e.task)}</td><td>${e.family}</td><td><span class="status-pill ${e.success?'':'failed'}">${e.success?'成功':'未成功'}</span></td><td>${e.policy_decisions}</td><td>${e.model_calls}</td><td>${e.elapsed_s.toFixed(1)} s</td><td>${reason[e.end_reason]||esc(e.end_reason)}</td><td><button class="text-button" data-task="${e.task}">回放 ↗</button></td></tr>`).join(''):'<tr><td colspan="8" class="empty">没有匹配的任务，请调整搜索词或结果筛选。</td></tr>';
  }
  ['search','status-filter','sort'].forEach(id=>$(id).addEventListener('input',table));table();
  const featured=[['stack_blocks_three','三层积木'],['handover_block','双臂交接'],['open_microwave','打开微波炉'],['place_empty_cup','杯子放置 · 未成功'],['scan_object','扫描物体 · 未成功']];
  $('featured').innerHTML=featured.map(([task,label])=>`<button data-task="${task}" aria-pressed="false">${label}</button>`).join('');
  document.addEventListener('click',event=>{const b=event.target.closest('[data-task]');if(b)selectTask(b.dataset.task,!b.closest('#featured'));});
  function cameras(frames,label){
    $('observation-label').textContent=label;
    $('cameras').innerHTML=[['head_camera','头部相机'],['left_camera','左腕相机'],['right_camera','右腕相机']].map(([key,name])=>`<figure>${frames[key]?`<img src="${media(selected.task,`frame/${frames[key]}`)}" alt="${name} · ${esc(label)}">`:'<p class="micro">无该相机记录</p>'}<figcaption>${name}</figcaption></figure>`).join('');
  }
  function step(index){
    currentStep=index;const t=selected.trace[index];if(!t)return;
    document.querySelectorAll('.trace-row').forEach((b,i)=>b.setAttribute('aria-pressed',String(i===index)));
    cameras(t.frames,`动作 ${index+1} 前观测 / observation ${t.observation_id}`);
    $('trace-detail').innerHTML=`<p><strong>${esc(t.skill)} · ${esc(t.arm||'—')}</strong></p><p>动作执行：${t.skill_success===true?'成功':t.skill_success===false?'未成功':'未记录'}；官方任务：${t.official_success===true?'成功':'尚未成功'}。</p>${t.failure?`<p><code>${esc(t.failure)}</code></p>`:''}${t.recovery?`<p>${esc(t.recovery)}</p>`:''}`;
  }
  function renderEpisode(e){
    $('episode-video').pause();selected=e;$('episode-title').textContent=e.task;$('episode-instruction').textContent=e.instruction;
    $('episode-badge').textContent=e.success?'官方成功':'官方未成功';$('episode-badge').className=`badge ${e.success?'':'failed'}`;
    $('episode-stats').textContent=`seed ${e.seed} · ${e.policy_decisions} 决策 · ${e.model_calls} 模型调用 · ${e.elapsed_s.toFixed(1)} s`;
    const video=$('episode-video');video.pause();$('media-error').hidden=true;
    video.poster=media(e.task,`frame/${e.trace[0]?.frames.head_camera||e.final_frames.head_camera}`);
    video.src=media(e.task,'video');video.load();$('video-link').href=video.src;
    $('trace-count').textContent=`${e.trace.length} 条动作记录`;
    $('trace-list').innerHTML=e.trace.map((t,i)=>`<button class="trace-row" data-step="${i}" aria-pressed="false"><span>${String(i+1).padStart(2,'0')}</span><span><strong>${esc(t.skill)} <small>${esc(t.arm||'')} ${esc(t.target||'')}${t.destination?` → ${esc(t.destination)}`:''}</small></strong></span><span class="trace-result ${t.skill_success===false?'failed':''}">${t.skill_success===true?'已执行':t.skill_success===false?'未成功':'未记录'}</span></button>`).join('');
    $('trace-list').scrollTop=0;step(0);
    document.querySelectorAll('#featured button').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.task===e.task)));
  }
  $('trace-list').addEventListener('click',event=>{const b=event.target.closest('[data-step]');if(b)step(Number(b.dataset.step));});
  $('final-observation').addEventListener('click',()=>{cameras(selected.final_frames,'任务结束后的最终观测');document.querySelectorAll('.trace-row').forEach(b=>b.setAttribute('aria-pressed','false'));$('trace-detail').innerHTML=`<p><strong>${reason[selected.end_reason]}</strong></p><p>官方结果：${selected.success?'成功':'未成功'}</p>${selected.error?`<p>${esc(selected.error)}</p>`:''}${selected.last_failure?`<p>最后动作反馈：${esc(selected.last_failure)}</p>`:''}`;});
  $('episode-video').addEventListener('error',()=>{$('media-error').hidden=false;});
  $('episode-video').addEventListener('loadedmetadata',()=>{try{const t=Number(localStorage.getItem(`v72-video-${selected.task}`));if(Number.isFinite(t)&&t>0&&t<$('episode-video').duration)$('episode-video').currentTime=t;}catch(_){}});
  $('episode-video').addEventListener('timeupdate',()=>{try{localStorage.setItem(`v72-video-${selected.task}`,String($('episode-video').currentTime));}catch(_){}});
  const failed=A.total-A.successes;
  $('termination-bars').innerHTML=Object.entries(A.end_reasons).filter(([key])=>key!=='environment_success').map(([key,n])=>`<div class="reason-row"><div class="bar-label"><span>${reason[key]}</span><b>${n} / ${failed} · ${pct(n,failed)}</b></div><div class="bar-track"><span style="width:${n/failed*100}%"></span></div></div>`).join('');
  $('failure-signals').innerHTML=D.failures.slice(0,5).map(f=>`<div class="signal-row"><span>${esc(f.message)}</span><b>${f.count} 次</b></div>`).join('');
  $('cost-insight').textContent=`成功任务平均 ${A.by_outcome.true.mean_decisions.toFixed(1)} 次决策、${A.by_outcome.true.mean_elapsed_s.toFixed(1)} 秒；未成功任务平均 ${A.by_outcome.false.mean_decisions.toFixed(1)} 次决策、${A.by_outcome.false.mean_elapsed_s.toFixed(1)} 秒。`;
  if(window.echarts){
    const chart=echarts.init($('cost-chart'));chart.setOption({animation:false,grid:{left:52,right:24,top:24,bottom:66},tooltip:{trigger:'item',formatter:p=>`${esc(p.data[2])}<br>${p.data[0]} 次决策 / ${p.data[1].toFixed(1)} 秒`},legend:{bottom:0,data:['成功','未成功']},xAxis:{type:'value',min:0,max:30,interval:5,axisLine:{lineStyle:{color:'#bbc1b5'}},splitLine:{lineStyle:{color:'#e5e7df'}}},yAxis:{type:'value',min:0,axisLabel:{color:'#6d736b'},splitLine:{lineStyle:{color:'#e5e7df'}}},series:[true,false].map(s=>({type:'scatter',name:s?'成功':'未成功',symbolSize:9,itemStyle:{color:s?'#3a7359':'#ab5144'},data:D.episodes.filter(e=>e.success===s).map(e=>[e.policy_decisions,e.elapsed_s,e.task])}))});
    chart.on('click',p=>selectTask(p.data[2],true));window.addEventListener('resize',()=>chart.resize());
  }else{$('cost-chart').innerHTML='<p class="micro">图表库未加载。精确决策数与耗时仍可在任务明细中筛选和排序。</p>';}
  const cases=[
    ['place_empty_cup','反复触发动作约束',`${D.episodes.find(e=>e.task==='place_empty_cup').trace.filter(t=>t.failure==='PUSH distance must be 1.5–45 cm').length} / 30 次动作返回 PUSH distance must be 1.5–45 cm，最终耗尽决策预算。记录显示相同约束反复触发；可据此检查目标定位与重规划是否有效改变了动作。`],
    ['scan_object','技能执行与目标达成存在差距','最后一次 present 动作 skill_success 为 true，官方 success 仍为 false。轨迹提示计划已执行完但官方目标未达成，最终耗尽 30 次决策。'],
    ['open_laptop','接触定位未通过有效性校验','记录 4 次执行动作后以 invalid_model_action 结束。result.error 记录了接触点落在机器人上，以及铰链端点缺少当前非机器人区域深度证据；定位修复后仍未通过校验。']
  ];
  $('failure-cases').innerHTML=cases.map(([task,title,description])=>{const e=D.episodes.find(e=>e.task===task);return `<article class="case-card"><img loading="lazy" src="${media(task,`frame/${e.final_frames.head_camera}`)}" alt="${esc(task)} 最终观测"><h3>${title}</h3><code>${task}</code><p>${description}</p><button class="text-button" data-task="${task}">查看失败证据 →</button></article>`;}).join('');
  $('recovery-note').innerHTML=`<strong>成功轨迹中的恢复也值得看。</strong> ${A.recovered_episodes} 个最终成功的任务曾记录动作失败反馈。全批次共记录 ${A.failed_actions} 次动作失败反馈；这些次数不能当作失败 episode 数。`;
  const names={'completion_review_protocol_standard_20260922':'5 任务试跑','full50_completion_review_protocol_standard_20260922':'初始 50 任务批次','full50_completion_review_restored_protocol_standard_20260922':'恢复协议后的完整批次 · 主结果'};
  $('cohort-table').innerHTML=`<table><thead><tr><th>批次</th><th>成功 / 请求数</th><th>已完成</th><th>运行错误</th><th>成功 / 请求数比例</th></tr></thead><tbody>${D.cohorts.map(c=>`<tr><td>${names[c.id]}<span class="cohort-id">${c.id}</span></td><td>${c.successes} / ${c.requested}</td><td>${c.completed}</td><td>${c.errors}</td><td>${pct(c.successes,c.requested)}</td></tr>`).join('')}</tbody></table><p class="micro">completed 包含成功及任务未成功；运行错误单独计数。初始批次有 11 个运行错误，不使用仅完成子集作为分母，也不将不同批次混合计分。</p>`;
  function showRepeat(r,live=false){
    const pending=r.requested-r.completed-r.errors;
    $('repeat-stats').textContent=`已完成 ${r.completed} / ${r.requested} · 成功 ${r.successes} · 未成功 ${r.failures} · 运行错误 ${r.errors} · 未完成 ${pending}${r.completed?` · 已完成子集成功率 ${pct(r.successes,r.completed)}`:' · 成功率尚不可用'}`;
    $('repeat-updated').textContent=`${live?'已加载发布快照':'发布快照'} · 数据时间 ${r.updated_utc}。这是发布时的静态快照，不实时连接评测服务器；未完成项不计为失败。`;
  }
  showRepeat(D.repeat);
  $('refresh-repeat').addEventListener('click',async()=>{const b=$('refresh-repeat');b.disabled=true;try{const response=await fetch('data/repeat.json',{cache:'no-store'});if(!response.ok)throw Error('http');showRepeat(await response.json(),true);}catch(_){$('repeat-updated').textContent='快照读取失败，仍显示页面内置数据。请检查网络后重试。';}finally{b.disabled=false;}});
  $('source-path').textContent=`${D.source}/${D.cohort}`;
  $('snapshot-time').textContent=`主批次更新时间 ${D.updated_utc} · 页面数据生成时间 ${D.generated_utc}`;
  const hashTask=location.hash.startsWith('#episode=')?decodeURIComponent(location.hash.slice(9)):null;
  renderEpisode(D.episodes.find(e=>e.task===hashTask)||initial);
})();
