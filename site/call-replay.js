(() => {
  'use strict';
  const host = document.getElementById('call-replay-app');
  if (!host) return;
  const esc = v => String(v ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const json = v => JSON.stringify(v, null, 2);
  const idOf = c => `call_${String(c.number).padStart(4, '0')}`;
  const promptOf = c => c.request.messages.map(m => typeof m.content === 'string' ? m.content : m.content.filter(p => p.type === 'text').map(p => p.text).join('\n')).join('\n\n');
  const seconds = c => Number(c.response?.elapsed_s ?? c.error?.elapsed_s) || 0;
  const label = {observe:'读取观测',request:'请求 VLM',response:'VLM 返回',reject:'Harness 拒绝',execute:'交给执行器',feedback:'环境反馈',end:'任务结束'};
  const code = (value, cls = '') => `<pre class="cr-code ${cls}">${esc(typeof value === 'string' ? value : json(value))}</pre>`;
  let catalogue = [], flow, events = [], cursor = 0, timer = null, delay = 2000, loading = 0;
  const cache = new Map();

  function buildEvents(d) {
    const list = [];
    d.steps.forEach((s, si) => {
      list.push({type:'observe',si});
      d.calls.filter(c => c.step === s.step).sort((a,b) => a.number - b.number).forEach(c => {
        list.push({type:'request',si,c}, {type:'response',si,c});
        if (c.rejection_feedback) list.push({type:'reject',si,c});
      });
      list.push({type:'execute',si}, {type:'feedback',si});
    });
    list.push({type:'end',si:d.steps.length-1});
    return list;
  }
  function counts() {
    const seen = events.slice(0, cursor+1);
    return {
      requests:seen.filter(e => e.type === 'request').length,
      decisions:seen.filter(e => e.type === 'feedback').length,
      elapsed:seen.filter(e => e.type === 'response').reduce((n,e) => n + seconds(e.c),0)
    };
  }
  function compactAction(a) {
    return Object.fromEntries(['skill','arm','target','delta','axis','angle','location','bbox','point'].filter(k => a[k] !== undefined).map(k => [k,a[k]]));
  }
  function description(e) {
    const s = flow.steps[e.si], n = flow.calls.filter(c => c.step === s.step).length;
    if (e.type === 'observe') return n ? `本决策归属 ${n} 次模型调用；执行器尚未收到本步动作。` : `本决策没有新模型调用，沿用已有计划或 harness 规则。`;
    if (e.type === 'request') return `${e.c.kind} · ${e.c.attachments.length ? `${e.c.attachments.length} 张实际输入图片` : '纯文本请求，无图片'}。本次请求本身不执行机器人动作。`;
    if (e.type === 'response') return e.c.rejection_feedback ? '这份输出随后被 harness 拒绝，没有直接执行。下一事件显示原始拒绝原因。' : '这是原始模型返回，不等于机器人已执行。计划、核验和定位输出仍要由 harness 处理。';
    if (e.type === 'reject') return `拒绝反馈写入下一次请求 ${idOf({number:e.c.rejection_feedback.next_call})}，机器人本步仍未执行。`;
    if (e.type === 'execute') return `前面的 ${n} 次调用汇入本决策的 ${s.action.skill}。${s.result.subactions?.length || 0} 个运动子步骤由 harness 执行，不各自算一次 VLM 调用。`;
    if (e.type === 'feedback') return s.result.success ? '官方环境已报告 success=true。' : '本步记录已写入轨迹；无论技能是否成功，官方任务目标仍未完成。';
    return flow.result.success ? '任务完成，由环境官方 success 判定。' : `任务未完成 · ${flow.result.end_reason}。技能执行无 failure 不等于完成任务。`;
  }
  function body(e) {
    const s = flow.steps[e.si];
    if (e.type === 'observe') return `<h4>本步真实状态</h4>${code({observation_id:s.observation.observation_id,endpose:s.observation.endpose,sensors:s.observation.sensors})}<p class="micro">这是 harness 保存的观测，不表示所有字段都发送给 VLM；实际输入以下一事件 request 为准。</p>`;
    if (e.type === 'request') return `<h4>实际 prompt · 完整文本</h4>${code(promptOf(e.c),'cr-prompt')}<a href="${esc(e.c.request_url)}" target="_blank" rel="noopener">原始 request JSON ↗</a>`;
    if (e.type === 'response') return `<div class="cr-response-meta"><span>请求墙钟耗时 ${seconds(e.c).toFixed(2)} s</span><span>${e.c.response?.response?.x_enable_thinking === false ? 'enable_thinking=false' : '思考模式未记录'}</span></div><h4>原始输出 · 未改写</h4>${code(e.c.raw_text || e.c.error?.error || '无完整响应','cr-output')}<details><summary>本次实际 prompt</summary>${code(promptOf(e.c))}</details>${e.c.response_url ? `<a href="${esc(e.c.response_url)}" target="_blank" rel="noopener">原始 response JSON ↗</a>` : ''}`;
    if (e.type === 'reject') return `<div class="cr-rejection"><h4>未执行此输出</h4>${code(e.c.rejection_feedback.reason,'cr-reason')}<p>下一次调用 ${idOf({number:e.c.rejection_feedback.next_call})} 会收到这段反馈。</p></div><details><summary>被拒绝的模型原文</summary>${code(e.c.raw_text)}</details>`;
    if (e.type === 'execute') return `<h4>最终动作参数</h4>${code(compactAction(s.action),'cr-action')}<h4>本次执行记录中的子步骤</h4><ol class="cr-subactions">${(s.result.subactions || []).map(x => `<li><code>${esc(x.name)}</code><span>${x.motion_ok === true ? '运动成功' : x.success === true ? '目标完成' : '见原始记录'}${x.position_error_m != null ? ` · 误差 ${(x.position_error_m*1000).toFixed(1)} mm` : ''}</span></li>`).join('') || '<li>没有已保存的运动子步骤</li>'}</ol><details><summary>完整 harness action 与 result</summary>${code({action:s.action,result:s.result})}</details>`;
    if (e.type === 'feedback') return `${code({skill_success:s.result.skill_success,official_success:s.result.success,failure:s.result.failure,sensors:s.result.sensors})}<p>决策完成计数在此增加 1；接下来再读取下一次观测。</p>`;
    return `<h4 class="${flow.result.success?'cr-success':'cr-failure'}">${flow.result.success?'任务完成':'任务未完成'}</h4>${code({success:flow.result.success,end_reason:flow.result.end_reason,policy_decisions:flow.result.policy_decisions,model_calls:flow.result.model_calls,elapsed_s:flow.result.elapsed_s})}<a href="${esc(flow.raw['result.json'])}" target="_blank" rel="noopener">原始结果 ↗</a>`;
  }
  function media(e) {
    const s = flow.steps[e.si];
    if (e.c) {
      if (!e.c.attachments.length) return `<div class="cr-text-only"><span>TEXT ONLY</span><h4>这次没有发送图片。</h4><p>右侧 prompt 是模型真正收到的文本。不会把三路相机观测冒充成本次输入。</p></div>`;
      return e.c.attachments.map(a => `<figure><a href="${esc(a.url)}" target="_blank" rel="noopener"><img src="${esc(a.url)}" alt="${esc(idOf(e.c))} 实际输入 ${esc(a.name)}"></a><figcaption>${esc(a.name)} · ${a.width} × ${a.height} · 原始输入 JPEG</figcaption></figure>`).join('');
    }
    const after = e.type === 'feedback' || e.type === 'end';
    const rgb = after ? (flow.steps[e.si+1]?.rgb || flow.final_rgb) : s.rgb;
    return `<figure class="cr-head-camera"><a href="${esc(rgb.head_camera)}" target="_blank" rel="noopener"><img src="${esc(rgb.head_camera)}" alt="${after?'动作后下一次':'动作前'}头部观测"></a><figcaption>${after?'动作后下一次保存的观测':'动作前保存的观测'} · 不是运动中的同步画面</figcaption></figure><div class="cr-wrists">${['left_camera','right_camera'].map(k=>`<figure><img src="${esc(rgb[k])}" alt="${k}"><figcaption>${k}</figcaption></figure>`).join('')}</div>`;
  }
  function save() {
    const value = {id:flow.id,event:cursor};
    history.replaceState(null,'',`#calls=${flow.id}&event=${cursor}`);
    try {localStorage.setItem('v72-call-replay',JSON.stringify(value));} catch (_) {}
  }
  function stop() {clearTimeout(timer);timer=null;const b=host.querySelector('[data-cr-play]');if(b){b.textContent='播放';b.setAttribute('aria-pressed','false');}}
  function go(n, persist=true) {cursor=Math.max(0,Math.min(events.length-1,Math.trunc(n)||0));renderEvent();if(persist)save();}
  function schedule() {timer=setTimeout(()=>{if(cursor>=events.length-1){stop();return;}go(cursor+1);if(cursor===events.length-1)stop();else schedule();},delay);}
  function shell() {
    host.innerHTML=`<div class="cr-choices">${catalogue.map(m=>`<button data-cr-flow="${esc(m.id)}" aria-pressed="${m.id===flow.id}"><small>${m.outcome==='success'?'成功记录':'失败记录'}</small><strong>${esc(m.task)}</strong><span>${m.calls} 次调用 / ${m.decisions} 次决策</span></button>`).join('')}</div><div class="cr-chapters"><button data-cr-chapter="first">第一步：多次询问，一次抓取</button><button data-cr-chapter="zero">零调用：沿用已有计划</button><button data-cr-chapter="reject" ${flow.calls.some(c=>c.rejection_feedback)?'':'disabled'}>拒绝后重新询问</button><button data-cr-chapter="end">最终判定</button></div><div class="cr-player"><div class="cr-heading"><div><strong>${esc(flow.result.task)}</strong><p>${esc(flow.result.instruction)}</p></div><span class="cr-recorded">RECORDED TRACE</span></div><div class="cr-counters" aria-live="polite"></div><div class="cr-controls"><button data-cr-play aria-pressed="false">播放</button><button data-cr-restart aria-label="从头开始">重播</button><button data-cr-prev aria-label="上一事件">←</button><button data-cr-next aria-label="下一事件">→</button><span class="cr-position"></span><input class="cr-scrubber" type="range" min="0" max="${events.length-1}" value="${cursor}" aria-label="调用事件位置"><label>阅读节奏<select data-cr-speed aria-label="阅读节奏"><option value="4000">慢 · 4 秒/事件</option><option value="2000">中 · 2 秒/事件</option><option value="800">快 · 0.8 秒/事件</option></select></label><button data-cr-share>复制链接</button></div><div class="cr-stage"><aside class="cr-observation"><h4 class="cr-media-title"></h4><div class="cr-media"></div><div class="cr-decision-note"></div><div class="cr-evidence"></div></aside><div class="cr-transcript"><div class="cr-feed" role="region" tabindex="0" aria-label="已发生的调用与执行事件"></div><article class="cr-detail"></article></div></div><div class="cr-strip" role="group" aria-label="逐决策调用数">${flow.steps.map((s,i)=>{const n=flow.calls.filter(c=>c.step===s.step).length;return `<button data-cr-step="${i}" title="决策 ${i+1}：${n} 次调用，${s.action.skill}"><small>D${i+1}</small><b>${n}</b><span>${esc(s.action.skill)}</span></button>`;}).join('')}</div><p class="cr-strip-key">每格是一条策略决策，数字是该决策内的模型调用数；宽度不代表耗时。点击任意格跳到本步开始。</p></div>`;
    host.querySelector('[data-cr-speed]').value=String(delay);
    host.setAttribute('aria-busy','false');renderEvent();
  }
  function renderEvent() {
    const e=events[cursor],s=flow.steps[e.si],c=counts(),cs=flow.calls.filter(x=>x.step===s.step);
    host.querySelector('.cr-counters').innerHTML=`<span><b data-cr-count="requests">${c.requests}</b> / ${flow.calls.length} 次模型请求</span><span><b data-cr-count="decisions">${c.decisions}</b> / ${flow.steps.length} 次已完成决策</span><span><b>${c.elapsed.toFixed(1)} s</b> 已返回请求累计耗时</span>`;
    host.querySelector('.cr-position').textContent=`事件 ${cursor+1} / ${events.length}`;
    host.querySelector('.cr-scrubber').value=cursor;
    host.querySelector('[data-cr-prev]').disabled=cursor===0;
    host.querySelector('[data-cr-next]').disabled=cursor===events.length-1;
    host.querySelector('.cr-media-title').textContent=e.c?`${idOf(e.c)} · 实际模型输入`:'环境观测 · harness 输入';
    host.querySelector('.cr-media').innerHTML=media(e);
    host.querySelector('.cr-decision-note').innerHTML=`<span>当前决策 ${e.si+1} / ${flow.steps.length} · source step ${s.step}</span><strong>${cs.length} 次调用 → ${esc(s.action.skill)}</strong><p>${e.type==='feedback'||e.type==='end'?'本步执行记录已完成。':e.type==='execute'?'此处才把最终动作交给执行器。':'本决策尚未交给执行器；连续询问不会增加已完成决策数。'}</p>`;
    host.querySelector('.cr-evidence').innerHTML=`<a href="#workflow=${flow.id}&step=${s.step}${e.c?`&call=${e.c.number}`:''}">查看本步完整传感器与执行证据 ↓</a><a href="data/workflows/${flow.id}/workflow.json" target="_blank" rel="noopener">完整 workflow JSON ↗</a>`;
    const feed=host.querySelector('.cr-feed');
    feed.innerHTML=events.slice(0,cursor+1).map((x,i)=>`<button class="cr-event cr-${x.type}" data-cr-event="${i}" aria-current="${i===cursor}"><small>D${x.si+1}</small><span>${label[x.type]}${x.c?` · ${idOf(x.c)}`:''}</span><em>${x.type==='request'?esc(x.c.kind):x.type==='feedback'?`success=${flow.steps[x.si].result.success===true}`:x.type==='response'?`${seconds(x.c).toFixed(2)} s`:''}</em></button>`).join('');
    feed.scrollTop=feed.scrollHeight;
    host.querySelector('.cr-detail').innerHTML=`<div class="cr-event-title cr-${e.type}"><span>${label[e.type]}</span><h3>${e.c?`${idOf(e.c)} · ${esc(e.c.kind)}`:e.type==='end'?'官方终局判定':`决策 ${e.si+1} · ${esc(s.action.skill)}`}</h3></div><p class="cr-description">${esc(description(e))}</p>${body(e)}`;
    host.querySelectorAll('[data-cr-step]').forEach(b=>b.setAttribute('aria-current',Number(b.dataset.crStep)===e.si?'true':'false'));
    const strip=host.querySelector('.cr-strip'),active=strip.querySelector('[aria-current="true"]');
    strip.scrollLeft=Math.max(0,active.offsetLeft-strip.offsetLeft-strip.clientWidth/2+active.clientWidth/2);
  }
  async function choose(id, event=0, persist=true) {
    const m=catalogue.find(m=>m.id===id);if(!m)return;
    stop();const token=++loading;host.setAttribute('aria-busy','true');
    try {
      if(!cache.has(id)){const r=await fetch(m.url);if(!r.ok)throw Error(`HTTP ${r.status}`);cache.set(id,await r.json());}
      if(token!==loading)return;
      flow=cache.get(id);events=buildEvents(flow);cursor=Math.max(0,Math.min(events.length-1,Math.trunc(Number(event))||0));shell();if(persist)save();
    } catch(e) {if(token!==loading)return;host.setAttribute('aria-busy','false');host.innerHTML=`<p class="cr-error">调用回放加载失败：${esc(e.message)}。<button data-cr-retry>重试</button></p>`;}
  }
  function hashState(){const p=new URLSearchParams(location.hash.slice(1));return {id:p.get('calls'),event:p.get('event')};}
  host.addEventListener('click',async ev=>{
    const b=ev.target.closest('button');if(!b)return;
    if(b.hasAttribute('data-cr-retry')){await init();return;}
    if(b.dataset.crFlow){await choose(b.dataset.crFlow);return;}
    if(b.hasAttribute('data-cr-play')){if(timer){stop();return;}if(cursor===events.length-1)go(0);b.textContent='暂停';b.setAttribute('aria-pressed','true');schedule();return;}
    if(b.hasAttribute('data-cr-share')){save();try{await navigator.clipboard.writeText(location.href);b.textContent='已复制';}catch(_){b.textContent='请复制地址栏';}return;}
    stop();
    if(b.hasAttribute('data-cr-prev'))go(cursor-1);
    if(b.hasAttribute('data-cr-next'))go(cursor+1);
    if(b.hasAttribute('data-cr-restart'))go(0);
    if(b.hasAttribute('data-cr-event'))go(Number(b.dataset.crEvent));
    if(b.hasAttribute('data-cr-step'))go(events.findIndex(e=>e.si===Number(b.dataset.crStep)));
    if(b.dataset.crChapter){
      const key=b.dataset.crChapter;
      if(key==='first')go(0);
      if(key==='end')go(events.length-1);
      if(key==='reject'){
        const rejected=flow.calls.find(c=>c.parsed_output?.skill==='scan'&&c.rejection_feedback)||flow.calls.find(c=>c.rejection_feedback);
        go(events.findIndex(e=>e.type==='response'&&e.c===rejected));
      }
      if(key==='zero')go(events.findIndex(e=>e.type==='observe'&&!flow.calls.some(c=>c.step===flow.steps[e.si].step)));
    }
  });
  host.addEventListener('input',e=>{if(e.target.matches('.cr-scrubber')){stop();go(Number(e.target.value));}});
  host.addEventListener('change',e=>{if(e.target.matches('[data-cr-speed]')){delay=Number(e.target.value);if(timer){clearTimeout(timer);schedule();}}});
  host.addEventListener('click',e=>{if(e.target.closest('a'))stop();});
  document.addEventListener('visibilitychange',()=>{if(document.hidden)stop();});
  window.addEventListener('resize',()=>{const feed=host.querySelector('.cr-feed');if(feed)feed.scrollTop=feed.scrollHeight;});
  window.addEventListener('hashchange',()=>{stop();const h=hashState();if(h.id&&catalogue.length)choose(h.id,h.event,false);});
  async function init(){
    try{const r=await fetch('data/workflows/index.json');if(!r.ok)throw Error(`HTTP ${r.status}`);catalogue=await r.json();
      let h=hashState();if(!h.id){try{h=JSON.parse(localStorage.getItem('v72-call-replay'))||h;}catch(_){}}
      await choose(catalogue.some(m=>m.id===h.id)?h.id:catalogue[0].id,h.event||0,false);
      if(location.hash==='#calls'||location.hash.startsWith('#calls='))window.scrollTo({top:document.getElementById('calls').getBoundingClientRect().top+window.scrollY-100});
    }catch(e){host.setAttribute('aria-busy','false');host.innerHTML=`<p class="cr-error">无法加载调用索引：${esc(e.message)}。<button data-cr-retry>重试</button></p>`;}
  }
  init();
})();
