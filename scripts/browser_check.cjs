/* Browser integration check. PLAYWRIGHT_MODULE can point to an existing install. */
const assert=require('node:assert/strict');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
(async()=>{
  const browser=await chromium.launch({headless:true,args:['--no-sandbox']});
  const page=await browser.newPage({viewport:{width:1440,height:1000}});
  const errors=[];page.on('pageerror',e=>errors.push(String(e)));
  const base=process.env.SITE_URL||'http://127.0.0.1:4187/';
  await page.goto(base);await page.waitForSelector('.trace-row');
  assert.match(await page.locator('#metrics').innerText(),/46.6%/);
  assert.equal(await page.locator('#task-matrix [data-episode]').count(),500);
  assert.equal(await page.locator('#task-rows tr').count(),50);
  await page.selectOption('#status-filter','zero');assert.equal(await page.locator('#task-rows tr').count(),9);
  await page.selectOption('#status-filter','perfect');assert.equal(await page.locator('#task-rows tr').count(),5);
  await page.selectOption('#status-filter','all');
  await page.locator('#search').fill('scan_object');assert.equal(await page.locator('#task-rows tr').count(),1);
  await page.locator('#task-rows [data-episode="scan_object__00"]').click();
  await page.waitForFunction(()=>document.querySelector('#trace-list').children.length===30);
  await page.locator('.trace-row').first().click();assert.match(await page.locator('#observation-label').innerText(),/动作 1/);
  await page.waitForFunction(()=>document.querySelector('#episode-video').readyState>=2);
  await page.locator('#episode-video').evaluate(v=>v.play());
  await page.waitForFunction(()=>document.querySelector('#episode-video').currentTime>0.1);
  await page.locator('#episode-video').evaluate(v=>v.pause());
  await page.selectOption('#playback-rate','2');assert.equal(await page.locator('#episode-video').evaluate(v=>v.playbackRate),2);
  await page.selectOption('#replay-task','place_empty_cup');
  await page.waitForFunction(()=>document.querySelector('#episode-title').textContent.includes('place_empty_cup'));
  assert.match(await page.locator('#episode-stats').innerText(),/精确值未记录/);
  await page.waitForFunction(()=>document.querySelector('#trace-list .trace-row'));
  await page.locator('#final-observation').click();assert.match(await page.locator('#observation-label').innerText(),/无最终观测/);
  await page.selectOption('#replay-repeat','place_empty_cup__01');
  await page.waitForFunction(()=>document.querySelector('#episode-badge').textContent==='官方成功');
  await page.goto(base+'#episode=open_laptop__00');await page.waitForSelector('.trace-row');
  assert.match(await page.locator('#episode-title').innerText(),/open_laptop/);
  assert.match(await page.locator('#episode-end').innerText(),/Hinge endpoints/);
  await page.locator('#analysis').scrollIntoViewIfNeeded();
  await page.locator('#failure-cases').scrollIntoViewIfNeeded();
  await page.waitForFunction(()=>[...document.querySelectorAll('#failure-cases img')].every(i=>i.complete&&i.naturalWidth>0));
  await page.locator('#failure-cases').screenshot({path:'/tmp/v72-official500-cases.png'});
  await page.setViewportSize({width:390,height:844});await page.goto(base);
  await page.waitForSelector('.trace-row');
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true,'mobile overflow');
  await page.screenshot({path:'/tmp/v72-official500-mobile.png'});
  // Check every original MP4 is decodable, not just that the file exists.
  const videoCheck=await page.evaluate(async()=>{
    const pending=[...window.REPORT_DATA.episodes],bad=[];let checked=0;
    async function worker(){while(pending.length){const e=pending.shift();const v=document.createElement('video');v.preload='metadata';
      await new Promise(resolve=>{const timer=setTimeout(()=>end('timeout'),20000);let finished=false;
        function end(error){if(finished)return;finished=true;clearTimeout(timer);if(error)bad.push([e.id,error]);checked++;v.removeAttribute('src');v.load();resolve();}
        v.onloadedmetadata=()=>end(v.duration>0?null:'empty');v.onerror=()=>end('decode');v.src=e.media+'/video.mp4';
      });}}
    await Promise.all(Array.from({length:4},worker));return {checked,bad};
  });
  assert.equal(videoCheck.checked,500);assert.deepEqual(videoCheck.bad,[]);assert.deepEqual(errors,[]);
  console.log(JSON.stringify({passed:true,videos:videoCheck.checked,checks:'500 matrix cells; filters; search; episode switching; trace; video playback; rate; timeout; deep links; bad-case images; mobile; no JS errors'}));
  await browser.close();
})().catch(e=>{console.error(e);process.exit(1)});
