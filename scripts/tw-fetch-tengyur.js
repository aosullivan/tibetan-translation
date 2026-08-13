(async () => {
  console.log('=== Tengyur XML fetch on', location.href, '===');
  if (!location.hostname.includes('tibetan.works')) { console.error('Run on a https://tibetan.works/ tab.'); return; }
  const COLLECTION = 'tengyur', WINDOW = 300, delayMs = 130;
  // tohoku match is primary; tib = title-substring fallback for pages lacking a tohoku tag
  const targets = [
    { toh: 4038, sanskrit: 'asanga-viniscayasamgrahani',      tib: 'རྣམ་པར་གཏན་ལ་དབབ་པ' },
    { toh: 2492, sanskrit: 'padmavajra-samayapanca',          tib: 'དམ་ཚིག་ལྔ་པ' },
    { toh: 4090, sanskrit: 'vasubandhu-abhidharmakosabhasya', tib: 'ཆོས་མངོན་པའི་མཛོད་ཀྱི་བཤད་པ' },
  ];
  const sleep = ms => new Promise(r => setTimeout(r, ms));
  const PUNC = ['་', '།', '༎', '༏', '༐', '༑'];
  const norm = s => { s = (s || '').replace(/\s+/g, ''); for (const p of PUNC) s = s.split(p).join(''); return s; };
  const TIB = /[ༀ-࿿]/;
  const deEnt = s => s.replace(/&#39;/g, "'").replace(/&amp;/g, '&').replace(/&#(\d+);/g, (m, n) => String.fromCharCode(+n));
  const u = i => 'https://tibetan.works/etext/reader.php?collection=' + COLLECTION + '&index=' + i;
  const got = [];
  const info = h => { const t = (h.match(/tohoku="(\d+)/) || [])[1]; const ti = [...h.matchAll(/<title>([\s\S]*?)<\/title>/g)].map(m => m[1].trim()); return { toh: t ? +t : null, title: ti.find(x => TIB.test(x)) || '' }; };
  const xmlOf = h => { let s = h.indexOf('<?xml-stylesheet'); if (s < 0) s = h.indexOf('<text>'); const e = h.lastIndexOf('</text>'); return (s >= 0 && e >= 0) ? h.slice(s, e + 7) : null; };
  const save = (h, i, t) => { const x = xmlOf(h); if (!x) return false; const fm = h.match(/value=['"]([^'"]+\.xml)['"]/i); const b = fm ? deEnt(fm[1]).split('/').pop() : ('toh' + t.toh + '_idx' + i + '.xml'); const n = 'tengyur-toh' + t.toh + '-' + t.sanskrit + '__' + b; const a = Object.assign(document.createElement('a'), { href: URL.createObjectURL(new Blob([x], { type: 'application/xml' })), download: n }); document.body.appendChild(a); a.click(); a.remove(); URL.revokeObjectURL(a.href); return n; };
  async function peek(i) { const r = await fetch(u(i), { credentials: 'include' }); if (!r.ok || !r.body) return { toh: null, title: '' }; const rd = r.body.getReader(), dec = new TextDecoder(); let h = ''; while (h.length < 6000) { const { done, value } = await rd.read(); if (done) break; h += dec.decode(value, { stream: true }); if (h.includes('</info>')) break; } try { await rd.cancel(); } catch (e) {} return info(h); }
  for (const t of targets) {
    const hookN = norm(t.tib);
    const hit = o => o.toh === t.toh || (o.title && norm(o.title).includes(hookN));
    let done = false;
    try { const h = await (await fetch(u(t.toh), { credentials: 'include' })).text(); const o = info(h); if (hit(o)) { if (save(h, t.toh, t)) { got.push({ toh: t.toh, index: t.toh, foundToh: o.toh, title: o.title.slice(0, 32) }); console.log('>>> Toh ' + t.toh + ' @ ' + t.toh + '  ' + o.title.slice(0, 32)); done = true; } } else console.log('index ' + t.toh + ' = Toh ' + o.toh + ' ("' + o.title.slice(0, 18) + '") - scanning for ' + t.toh); } catch (e) { console.warn(t.toh + ' direct: ' + e.message); }
    for (let d = 1; d <= WINDOW && !done; d++) for (const i of [t.toh + d, t.toh - d]) { if (done || i < 1) continue; try { const o = await peek(i); if (hit(o)) { const h = await (await fetch(u(i), { credentials: 'include' })).text(); if (save(h, i, t)) { got.push({ toh: t.toh, index: i, foundToh: o.toh, title: o.title.slice(0, 32) }); console.log('>>> Toh ' + t.toh + ' @ ' + i + '  ' + o.title.slice(0, 32)); done = true; } } } catch (e) {} await sleep(delayMs); }
    if (!done) console.warn('NOT FOUND: Toh ' + t.toh + ' (' + t.sanskrit + ')');
  }
  console.log('=== DONE ==='); console.table(got);
})();
