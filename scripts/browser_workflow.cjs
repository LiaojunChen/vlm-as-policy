const assert=require('node:assert/strict');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
(async()=>{
  const b=await chromium.launch({headless:true,args:['--no-sandbox']});
  const p=await b.newPage({viewport:{width:1440,height:1000}}),errors=[];
  p.on('pageerror',e=>errors.push(String(e)));
  const base=process.env.SITE_URL||'http://127.0.0.1:4187/';
  await p.goto(base+'#workflow');await p.waitForSelector('.flow-step');
  let checkedCalls=0,checkedSteps=0;
  for(const [id,n] of [['stack_blocks_three__00',8],['scan_object__00',30]]){
    await p.locator(`[data-flow="${id}"]`).click();
    await p.waitForFunction(n=>document.querySelectorAll('.flow-step').length===n,n);
    const data=await p.evaluate(async id=>(await fetch(`data/workflows/${id}/workflow.json`)).json(),id);
    for(let i=0;i<n;i++){
      await p.locator(`.flow-step[data-step="${i}"]`).click();
      assert.match(await p.locator('.flow-step-head h3').innerText(),new RegExp(`Step ${i} ·`));
      assert.equal(await p.locator('.flow-motion tbody tr').count(),data.steps[i].result.subactions?.length||0);
      const calls=data.calls.filter(c=>c.step===i);
      assert.equal(await p.locator('.flow-call-list button').count(),calls.length);
      for(let j=0;j<calls.length;j++){
        await p.locator(`.flow-call-list button[data-call="${j}"]`).click();
        assert.equal(await p.locator('.flow-raw').textContent(),calls[j].raw_text);
        const prompt=calls[j].request.messages.map(m=>typeof m.content==='string'?m.content:m.content.filter(x=>x.type==='text').map(x=>x.text).join('\n')).join('\n\n');
        assert.equal(await p.locator('.flow-prompt').textContent(),prompt);
        assert.equal(await p.locator('.flow-model-images img').count(),calls[j].attachments.length);
        checkedCalls++;
      }
      assert.equal(await p.locator('#workflow-app a[href="undefined"]').count(),0);
      assert.doesNotMatch(await p.locator('body').textContent(),/robodawn/i);
      assert.equal(await p.locator('a[href*="robodawn"]').count(),0);
      checkedSteps++;
    }
    assert.match(await p.locator('.flow-end-title').innerText(),id.startsWith('stack')?/任务完成/:/任务未完成/);
  }
  await p.goto(base+'#workflow=scan_object__00&step=8&call=7');
  await p.waitForSelector('.flow-raw');
  assert.match(await p.locator('.flow-raw').textContent(),/"skill":"scan"/);
  assert.match(await p.locator('#workflow-app').innerText(),/Unknown skill; use the available skill names/);
  await p.locator('[data-camera="height"]').click();
  assert.equal(await p.locator('.flow-cameras img[src*="height.webp"]').count(),3);
  const validImages=await p.locator('#workflow-app img').evaluateAll(async imgs=>{
    await Promise.all(imgs.map(i=>i.decode()));return imgs.every(i=>i.naturalWidth>0);
  });assert(validImages);
  await p.locator('.flow-card').nth(1).scrollIntoViewIfNeeded();
  await p.screenshot({path:'/tmp/workflow-prompt-output.png'});
  await p.setViewportSize({width:390,height:844});await p.locator('#workflow').scrollIntoViewIfNeeded();
  assert(await p.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
  await p.screenshot({path:'/tmp/workflow-mobile-final.png'});
  await p.locator('[data-flow="stack_blocks_three__00"]').click();await p.waitForFunction(()=>document.querySelectorAll('.flow-step').length===8);
  await p.locator('[data-call="1"]').click();assert.match(await p.locator('#workflow-app').innerText(),/本次为纯文本请求/);
  await p.locator('.flow-step[data-step="2"]').click();assert.match(await p.locator('#workflow-app').innerText(),/本步没有模型调用/);
  assert.deepEqual(errors,[]);console.log(JSON.stringify({passed:true,decisions:checkedSteps,calls:checkedCalls,checks:'verbatim prompts + outputs, exact image attachments, subactions, success/failure ending, rejected scan, pure text, inherited plan, height maps, deep link, mobile, no JS errors'}));await b.close();
})().catch(e=>{console.error(e);process.exit(1)});
