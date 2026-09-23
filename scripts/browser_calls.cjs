const assert=require('node:assert/strict');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
(async()=>{
  const browser=await chromium.launch({headless:true,args:['--no-sandbox']});
  const page=await browser.newPage({viewport:{width:1440,height:1000}}),errors=[];
  page.on('pageerror',e=>errors.push(String(e)));
  const base=process.env.SITE_URL||'http://127.0.0.1:4187/';
  await page.goto(base+'#calls');await page.waitForSelector('.cr-player');
  let checked=0;
  for(const [id,n,steps] of [['stack_blocks_three__00',10,8],['scan_object__00',37,30]]){
    await page.locator('[data-cr-flow="'+id+'"]').click();
    await page.waitForFunction(s=>document.querySelectorAll('.cr-strip button').length===s,steps);
    const d=await page.evaluate(async id=>(await fetch('data/workflows/'+id+'/workflow.json')).json(),id);
    const seq=[];d.steps.forEach((s,si)=>{seq.push({type:'observe',si});d.calls.filter(c=>c.step===s.step).forEach(c=>{seq.push({type:'request',si,c},{type:'response',si,c});if(c.rejection_feedback)seq.push({type:'reject',si,c});});seq.push({type:'execute',si},{type:'feedback',si});});seq.push({type:'end',si:steps-1});
    assert.equal(Number(await page.locator('.cr-scrubber').getAttribute('max')),seq.length-1);
    let requests=0,decisions=0;
    for(let i=0;i<seq.length;i++){
      const e=seq[i];if(i)await page.locator('.cr-scrubber').fill(String(i));
      if(e.type==='request')requests++;if(e.type==='feedback')decisions++;
      assert.equal(await page.locator('[data-cr-count="requests"]').textContent(),String(requests));
      assert.equal(await page.locator('[data-cr-count="decisions"]').textContent(),String(decisions));
      if(e.type==='request'){
        const prompt=e.c.request.messages.map(m=>typeof m.content==='string'?m.content:m.content.filter(p=>p.type==='text').map(p=>p.text).join('\n')).join('\n\n');
        assert.equal(await page.locator('.cr-prompt').textContent(),prompt);
        assert.deepEqual(await page.locator('.cr-media img').evaluateAll(imgs=>imgs.map(i=>i.getAttribute('src'))),e.c.attachments.map(a=>a.url));
        if(!e.c.attachments.length)assert.match(await page.locator('.cr-text-only').innerText(),/没有发送图片/);
      }
      if(e.type==='response'){assert.equal(await page.locator('.cr-output').textContent(),e.c.raw_text);checked++;}
      if(e.type==='reject')assert.equal(await page.locator('.cr-reason').textContent(),e.c.rejection_feedback.reason);
      if(e.type==='execute')assert.equal(await page.locator('.cr-subactions li').count(),d.steps[e.si].result.subactions.length||1);
    }
    assert.equal(requests,n);assert.equal(decisions,steps);
    assert.match(await page.locator('.cr-detail').innerText(),id.startsWith('stack')?/任务完成/:/任务未完成/);
    await page.locator('[data-cr-chapter="zero"]').click();assert.match(await page.locator('.cr-description').innerText(),/没有新模型调用/);
    if(id.startsWith('scan')){
      await page.locator('[data-cr-chapter="reject"]').click();assert.match(await page.locator('.cr-output').textContent(),/"skill":"scan"/);
      await page.locator('[data-cr-next]').click();assert.match(await page.locator('.cr-reason').textContent(),/Unknown skill/);
      await page.locator('[data-cr-next]').click();assert.match(await page.locator('.cr-prompt').textContent(),/Unknown skill/);
      const link=page.url();await page.reload();await page.waitForSelector('.cr-prompt');assert.equal(page.url(),link);assert.match(await page.locator('.cr-prompt').textContent(),/Unknown skill/);
    }
  }
  await page.locator('[data-cr-chapter="first"]').click();await page.locator('[data-cr-speed]').selectOption('800');await page.locator('[data-cr-play]').click();
  await page.waitForFunction(()=>Number(document.querySelector('.cr-scrubber').value)>0);
  await page.locator('[data-cr-play]').click();assert.equal(await page.locator('[data-cr-play]').textContent(),'播放');
  await page.locator('[data-cr-chapter="reject"]').click();await page.locator('[data-cr-next]').click();
  await page.locator('.cr-player').screenshot({path:'/tmp/call-replay-desktop.png'});
  await page.setViewportSize({width:390,height:844});assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),'mobile overflow');
  await page.locator('.cr-player').screenshot({path:'/tmp/call-replay-mobile.png'});
  assert.deepEqual(errors,[]);console.log(JSON.stringify({passed:true,calls:checked,checks:'all requests and responses verbatim; exact attachments; all event counters; rejected scan repair; zero-call decisions; endings; playback; deep-link reload; mobile; no JS errors'}));await browser.close();
})().catch(e=>{console.error(e);process.exit(1)});
