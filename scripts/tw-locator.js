(async () => {
  console.log('=== tibetan.works LOCATOR (reports only, no downloads) ===');
  if (!location.hostname.includes('tibetan.works')) { console.error('Run on a https://tibetan.works/ tab.'); return; }
  const delayMs = 110;
  // each query scans its collection/range and reports the first title match (index, folio range, filename)
  const queries = [
    { label: 'Toh8 Satasahasrika',      collection: 'kangyur', hooks: ['སྟོང་ཕྲག་བརྒྱ་པ'],                   start: 1,    end: 80 },
    { label: 'Toh287 Saddharmasmrty',   collection: 'kangyur', hooks: ['དྲན་པ་ཉེ་བར་གཞག'],                  start: 200,  end: 1000 },
    { label: 'dbang gi yon tan rim pa', collection: 'tengyur', hooks: ['ཡོན་ཏན་གྱི་རིམ་པ', 'དབང་གི་ཡོན་ཏན'], start: 1330, end: 1520 },
  ];
  const sleep = ms => new Promise(r => setTimeout(r, ms));
  const PUNC = ['་', '།', '༎', '༏', '༐', '༑'];
  const norm = s => { s = (s || '').replace(/\s+/g, ''); for (const p of PUNC) s = s.split(p).join(''); return s; };
  const TIB = /[ༀ-࿿]/;
  const deEnt = s => s.replace(/&#39;/g, "'").replace(/&amp;/g, '&').replace(/&#(\d+);/g, (m, n) => String.fromCharCode(+n));
  const u = (c, i) => 'https://tibetan.works/etext/reader.php?collection=' + c + '&index=' + i;
  const found = [];
  async function peek(c, i) {
    const r = await fetch(u(c, i), { credentials: 'include' });
    if (!r.ok || !r.body) return null;
    const rd = r.body.getReader(), dec = new TextDecoder(); let h = '';
    while (h.length < 9000) { const { done, value } = await rd.read(); if (done) break; h += dec.decode(value, { stream: true }); if (h.includes('</info>')) break; }
    try { await rd.cancel(); } catch (e) {}
    const titles = [...h.matchAll(/<title>([\s\S]*?)<\/title>/g)].map(m => m[1].trim());
    const title = titles.find(x => TIB.test(x)) || '';
    const toh = (h.match(/tohoku="(\d+)/) || [])[1] || '';
    const rng = (h.match(/<range start="(\d+)" end="(\d+)"/) || []);
    const fm = h.match(/value=['"]([^'"]+\.xml)['"]/i);
    const file = fm ? deEnt(fm[1]).split('/').pop() : '';
    return { title, toh, range: rng[1] ? (rng[1] + '-' + rng[2]) : '', file };
  }
  for (const q of queries) {
    const hooks = q.hooks.map(norm);
    let hit = false;
    console.log('-- scanning ' + q.label + ' (' + q.collection + ' ' + q.start + '..' + q.end + ')');
    for (let i = q.start; i <= q.end && !hit; i++) {
      try {
        const o = await peek(q.collection, i);
        if (i % 25 === 0) console.log('   ...' + i);
        if (o && o.title && hooks.some(hk => norm(o.title).includes(hk))) {
          found.push({ query: q.label, collection: q.collection, index: i, toh: o.toh, range: o.range, title: o.title.slice(0, 40), file: o.file });
          console.log('FOUND ' + q.label + ' @ ' + q.collection + ' index ' + i + '  ' + o.title.slice(0, 40) + '  [' + o.file + ']');
          hit = true;
        }
      } catch (e) {}
      await sleep(delayMs);
    }
    if (!hit) console.warn('not located: ' + q.label + ' (widen range)');
  }
  console.log('=== LOCATOR DONE ==='); console.table(found);
})();
