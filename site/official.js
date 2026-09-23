(() => {
  'use strict';
  const D=window.REPORT_DATA, A=D.aggregate, $=id=>document.getElementById(id);
  const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const pct=(n,d)=>`${(n/d*100).toFixed(1)}%`;
  const reasons={environment_success:'官方判定成功',decision_budget:'决策预算耗尽',invalid_model_action:'动作无效终止',wall_time_budget:'墙钟预算耗尽'};
  const byId=new Map(D.episodes.map(e=>[e.id,e]));
  const media=(e,file)=>`${e.media}/${file}`;
  const poster=e=>media(e,`frame/${e.final_frames.head_camera||e.first_frames.head_camera}`);
  const initial=D.episodes.find(e=>e.task==='stack_blocks_three'&&e.success)||D.episodes[0];
  const charts=[];
  function chart(id,option){if(!window.echarts){$(id).textContent='图表未加载；精确数据可下载 CSV 查看。';return;}const c=echarts.init($(id));c.setOption({animation:false,...option});charts.push(c);return c;}
  window.addEventListener('resize',()=>charts.forEach(c=>c.resize()));
  $('hero-image').src=poster(initial);$('hero-link').href=media(initial,'video.mp4');
  $('metrics').innerHTML=[
    [pct(A.successes,A.total),'最终官方成功率',`${A.successes} / ${A.total} · 267 次未成功`],
    ['50 × 10','全部评测已完成','500 个 episode · 0 基础设施错误'],
    [A.mean_decisions.toFixed(2),'平均策略决策数',`中位数 ${A.median_decisions} · 上限 30`],
    [`${A.mean_elapsed_s.toFixed(1)} s`,'有记录样本平均耗时',`${A.elapsed_count}/500 有精确值 · ${A.elapsed_missing} 次超时缺失`]
  ].map(([n,l,s])=>`<div class="metric"><strong>${n}</strong><span>${l}</span><small>${s}</small></div>`).join('');
  const trial=(e,cls='trial')=>`<a class="${cls} ${e.success?'':'failed'}" href="${media(e,'video.mp4')}" target="_blank" rel="noopener" title="${esc(e.task)} · 第 ${e.repeat_index+1} 次 · seed ${e.seed} · ${e.success?'成功':reasons[e.end_reason]}" aria-label="${esc(e.task)} 第 ${e.repeat_index+1} 次，seed ${e.seed}，${e.success?'成功':'未成功'}，打开原始录像">${e.repeat_index+1}</a>`;
  $('task-matrix').innerHTML=`<div class="matrix-row matrix-head"><span>任务 / 接受序号 →</span>${Array.from({length:10},(_,i)=>`<span>${i+1}</span>`).join('')}<span>成功</span></div>`+D.tasks.map(t=>`<div class="matrix-row"><span class="matrix-label" title="${esc(t.task)}">${esc(t.task)}</span>${t.episodes.map(id=>trial(byId.get(id),'task-cell')).join('')}<b>${t.successes}/10</b></div>`).join('');
  $('family-bars').innerHTML=D.groups.map(g=>`<div class="family-row"><div class="bar-label"><span>${esc(g.name)}</span><b>${g.successes}/${g.total} · ${pct(g.successes,g.total)}</b></div><div class="bar-track"><span style="width:${g.successes/g.total*100}%"></span></div></div>`).join('');
  $('outcome-insight').innerHTML=`<strong>${A.perfect_tasks} 个任务 10/10 成功，${A.zero_tasks} 个任务 0/10 成功。</strong><br>其余 ${50-A.perfect_tasks-A.zero_tasks} 个任务的结果随 seed 变化。单次成功不能代表跨场景稳定性；每任务 10 次仍是有限样本。`;
  chart('task-distribution',{grid:{left:34,right:12,top:15,bottom:40},tooltip:{trigger:'axis'},xAxis:{type:'category',name:'成功次数 / 10',nameLocation:'middle',nameGap:26,data:Array.from({length:11},(_,i)=>i)},yAxis:{type:'value',minInterval:1},series:[{name:'任务数',type:'bar',itemStyle:{color:'#3a7359'},data:Array.from({length:11},(_,i)=>D.tasks.filter(t=>t.successes===i).length)}]});
  function table(){
    const term=$('search').value.toLowerCase().trim(), status=$('status-filter').value, sort=$('sort').value;
    const rows=D.tasks.filter(t=>`${t.task} ${t.family}`.toLowerCase().includes(term)&&(status==='all'||status==='perfect'&&t.successes===10||status==='zero'&&t.successes===0||status==='mixed'&&t.successes>0&&t.successes<10));
    rows.sort((a,b)=>sort==='task'?a.task.localeCompare(b.task):b[sort]-a[sort]||a.task.localeCompare(b.task));
    $('row-count').textContent=`显示 ${rows.length} / 50 个任务；每项统计均包含全部 10 次测试。`;
    $('task-rows').innerHTML=rows.length?rows.map(t=>`<tr><td class="task-name">${esc(t.task)}</td><td>${esc(t.family)}</td><td><span class="status-pill ${t.successes===0?'failed':''}">${t.successes}/10 · ${t.successes*10}%</span></td><td>${t.mean_decisions.toFixed(1)}</td><td>${t.mean_elapsed_s.toFixed(1)} s <small>(${t.elapsed_count}/10 有记录)</small></td><td><div class="mini-trials">${t.episodes.map(id=>trial(byId.get(id))).join('')}</div></td></tr>`).join(''):'<tr><td colspan="6" class="empty">没有匹配的任务。</td></tr>';
  }
  ['search','status-filter','sort'].forEach(id=>$(id).addEventListener('input',table));table();
  const failed=A.total-A.successes;
  $('termination-bars').innerHTML=Object.entries(A.end_reasons).filter(([k])=>k!=='environment_success').sort((a,b)=>b[1]-a[1]).map(([k,n])=>`<div class="reason-row"><div class="bar-label"><span>${reasons[k]}</span><b>${n} / ${failed} · ${pct(n,failed)}</b></div><div class="bar-track"><span style="width:${n/failed*100}%"></span></div></div>`).join('');
  $('failure-signals').innerHTML=D.failures.slice(0,7).map(f=>`<div class="signal-row"><span>${esc(f.message)}</span><b>${f.count} 次</b></div>`).join('');
  $('cost-insight').textContent=`成功 episode 平均 ${A.by_outcome.true.mean_decisions.toFixed(1)} 次决策、${A.by_outcome.true.mean_elapsed_s.toFixed(1)} 秒（耗时 n=${A.by_outcome.true.elapsed_count}）；失败 episode 平均 ${A.by_outcome.false.mean_decisions.toFixed(1)} 次决策、${A.by_outcome.false.mean_elapsed_s.toFixed(1)} 秒（耗时 n=${A.by_outcome.false.elapsed_count}）。散点和耗时均值仅含 ${A.elapsed_count} 个有精确 elapsed_s 的样本；${A.elapsed_missing} 次超时没有精确耗时，不能忽略它们推导全体平均速度。点击散点可回放。`;
  const scatter=chart('cost-chart',{grid:{left:55,right:20,top:30,bottom:65},tooltip:{trigger:'item',formatter:p=>`${esc(p.data[2])}<br>${p.data[0]} 次决策 / ${p.data[1].toFixed(1)} s`},legend:{bottom:0,data:['成功','未成功']},xAxis:{type:'value',min:0,max:30,name:'决策数',nameLocation:'middle',nameGap:25},yAxis:{type:'value',name:'墙钟 / 秒'},series:[true,false].map(s=>({type:'scatter',name:s?'成功':'未成功',symbolSize:7,itemStyle:{color:s?'#3a7359':'#ab5144',opacity:.65},data:D.episodes.filter(e=>e.success===s&&e.elapsed_s!==null).map(e=>[e.policy_decisions,e.elapsed_s,e.id])}))});
  if(scatter)scatter.on('click',p=>{const e=byId.get(p.data[2]);if(e)window.open(media(e,'video.mp4'),'_blank','noopener');});
  chart('decision-chart',{grid:{left:40,right:15,top:15,bottom:50},tooltip:{trigger:'axis'},legend:{bottom:0},xAxis:{type:'category',data:A.decision_bins.map(b=>b.label)},yAxis:{type:'value'},series:[true,false].map(s=>({name:s?'成功':'未成功',type:'bar',itemStyle:{color:s?'#3a7359':'#ab5144'},data:A.decision_bins.map(b=>b[String(s)])}))});
  $('failure-cases').innerHTML=D.cases.map((c,i)=>{const e=byId.get(c.id),task=D.tasks.find(t=>t.task===e.task),comparison=c.comparison_id?byId.get(c.comparison_id):null;return `<article class="case-card"><a href="${media(e,'video.mp4')}" target="_blank" rel="noopener"><img loading="lazy" src="${poster(e)}" alt="${esc(e.task)} 第 ${e.repeat_index+1} 次 · ${e.has_final_observation?'最终观测':'最后保存的动作前观测'}"></a><p class="eyebrow">CASE ${String(i+1).padStart(2,'0')} / ${esc(reasons[e.end_reason])}</p><h3>${esc(c.title)}</h3><code>${esc(e.task)} · seed ${e.seed}</code><div class="evidence-tag">第 ${e.repeat_index+1} 次 / 本任务 ${task.successes}/10 成功</div><p class="case-label">已记录的事实</p><p>${esc(c.observation)}</p><p class="case-label">诊断解释</p><p>${esc(c.interpretation)}</p><p class="case-label">待验证的改进</p><p>${esc(c.next_step)}</p><details><summary>原始反馈与证据位置</summary><p><code>${esc(c.signal||'全部动作 failure 为空；全部官方 success=false')}</code></p>${c.matching_steps.length?`<p>匹配 ${c.matching_steps.length} 条，source step：${c.matching_steps.join(', ')}</p>`:''}${e.error?`<p><code>${esc(e.error)}</code></p>`:''}<a href="${e.trace_url}" target="_blank" rel="noopener">本轮 trace JSON ↗</a></details><div class="case-actions"><a class="text-button" href="${media(e,'video.mp4')}" target="_blank" rel="noopener">打开失败录像 →</a>${comparison?`<a class="text-button" href="${media(comparison,'video.mp4')}" target="_blank" rel="noopener">打开同任务成功 seed →</a>`:'<span class="micro">本任务暂无成功样本</span>'}</div></article>`;}).join('');
  $('recovery-note').innerHTML=`<strong>成功过程也并非总是顺利。</strong> ${A.recovered_episodes} 个最终成功的 episode 曾出现动作失败反馈。全批次记录 ${A.failed_actions} 条动作失败反馈，不能当作失败 episode 数。同任务成功/失败回放是不同 seed 的观察对照，不是控制变量实验。`;
  $('source-path').textContent=`VLM as Policy v72 · ${D.cohort}`;
  $('snapshot-time').textContent=`评测开始 ${D.started_utc} · 完成 ${D.updated_utc} · 发布数据生成 ${D.generated_utc}`;
})();
