/* Browser integration check. PLAYWRIGHT_MODULE can point to an existing install. */
const assert=require('node:assert/strict');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
(async()=>{
  const browser=await chromium.launch({headless:true,args:['--no-sandbox']});
  const page=await browser.newPage({viewport:{width:1440,height:1000}});
  const errors=[];page.on('pageerror',e=>errors.push(String(e)));
  const base=process.env.SITE_URL||'http://127.0.0.1:4187/';
  await page.goto(base);await page.waitForSelector('#task-matrix a');
  assert.equal((await page.locator('h1').textContent()).replace(/\s/g,''),'9B的小模型在好的harness下可以比肩5T的大模型。');
  assert.match(await page.locator('#metrics').innerText(),/46.6%/);
  assert.match(await page.locator('.comparison').innerText(),/GPT-6.*zero-shot/);
  assert.deepEqual(await page.locator('.comparison-row strong').allTextContents(),['46.6%','53.2%']);
  assert.deepEqual(await page.locator('.comparison-row b').evaluateAll(b=>b.map(x=>x.style.width)),['46.6%','53.2%']);
  assert.match(await page.locator('.comparison').innerText(),/6.6 个百分点/);
  assert.equal(await page.locator('#replay, #episode-video, a[href="#replay"]').count(),0);
  assert.equal(await page.locator('#call-replay-title').textContent(),'操作过程回放');
  assert.equal(await page.locator('#workflow h2').textContent(),'harness详尽过程');
  assert.doesNotMatch(await page.locator('body').textContent(),/robodawn|看执行过程，也看失败现场|问了五次，才动一次|从传感器输入，到任务的结局/i);
  assert.equal(await page.locator('a[href*="robodawn"]').count(),0);
  assert.equal(await page.locator('#task-matrix a.task-cell').count(),500);
  assert.equal(await page.locator('#task-rows tr').count(),50);
  assert(await page.evaluate(()=>[...document.querySelectorAll('nav a')].every(a=>document.querySelector(a.hash))));
  await page.screenshot({path:'/tmp/v72-home-desktop.png'});
  await page.selectOption('#status-filter','zero');assert.equal(await page.locator('#task-rows tr').count(),9);
  await page.selectOption('#status-filter','perfect');assert.equal(await page.locator('#task-rows tr').count(),5);
  await page.selectOption('#status-filter','all');
  await page.locator('#search').fill('scan_object');assert.equal(await page.locator('#task-rows tr').count(),1);
  const link=page.locator('#task-rows a.trial').first();
  assert.match(await link.getAttribute('href'),/scan_object.*\/video.mp4$/);
  assert.equal(await link.getAttribute('target'),'_blank');
  // Exercise a real new-tab recording link and the browser's native player.
  const popupPromise=page.waitForEvent('popup');await link.click();const popup=await popupPromise;
  await popup.waitForSelector('video');await popup.waitForFunction(()=>document.querySelector('video').readyState>=2);
  await popup.locator('video').evaluate(v=>v.play());
  await popup.waitForFunction(()=>document.querySelector('video').currentTime>0.1);await popup.close();
  await page.locator('#failure-cases').scrollIntoViewIfNeeded();
  await page.locator('#failure-cases img').evaluateAll(imgs=>Promise.all(imgs.map(i=>i.decode())));
  assert(await page.locator('#failure-cases .case-actions a[href$="video.mp4"]').count()>0);
  for(const width of [320,390,768,1024,1440]){
    await page.setViewportSize({width,height:844});
    assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),'overflow at '+width+'px');
  }
  await page.setViewportSize({width:390,height:844});await page.goto(base);
  await page.waitForSelector('#task-matrix a');await page.screenshot({path:'/tmp/v72-home-mobile.png'});
  // Opt in to re-decoding all originals when media changes.
  if(process.env.CHECK_ALL_VIDEOS==='1'){
    const result=await page.evaluate(async()=>{
      const pending=[...window.REPORT_DATA.episodes],bad=[];let checked=0;
      async function worker(){while(pending.length){const e=pending.shift(),v=document.createElement('video');v.preload='metadata';
        await new Promise(resolve=>{let finished=false;const timer=setTimeout(()=>end('timeout'),20000);
          function end(error){if(finished)return;finished=true;clearTimeout(timer);if(error)bad.push([e.id,error]);checked++;v.removeAttribute('src');v.load();resolve();}
          v.onloadedmetadata=()=>end(v.duration>0?null:'empty');v.onerror=()=>end('decode');v.src=e.media+'/video.mp4';
        });}}
      await Promise.all(Array.from({length:4},worker));return {checked,bad};
    });assert.equal(result.checked,500);assert.deepEqual(result.bad,[]);
  }
  assert.deepEqual(errors,[]);
  console.log(JSON.stringify({passed:true,checks:'headlines; comparison; removed section; 500 links; filters; search; original video playback; cases; 5 viewport sizes; no JS errors'}));
  await browser.close();
})().catch(e=>{console.error(e);process.exit(1)});
