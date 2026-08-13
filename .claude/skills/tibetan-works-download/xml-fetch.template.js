/* =============================================================================
 * tibetan.works XML fetcher  (browser-console, run ON a https://tibetan.works tab)
 * -----------------------------------------------------------------------------
 * The assistant fills the TARGETS block and COLLECTION below for the requested
 * texts, then hands the user the finished snippet. This file is the canonical
 * template — keep the logic in sync with what gets generated.
 *
 * HOW IT WORKS (why this approach):
 *   reader.php?collection=<kangyur|tengyur>&index=<N> returns a JS shell with the
 *   FULL <pageset> XML embedded inside <div class='text'>. So we:
 *     1. stream just the first ~6 KB of each index to read <info><title> (cheap),
 *     2. on a Tibetan-title match, fetch the whole page and slice out <text>…</text>,
 *     3. save it as a real .xml (named from the page's own download-form filename).
 *   `index` is the Lhasa "H" number (kangyur) / global index (tengyur) — it is NOT
 *   the Toh number, so we always match by Tibetan title, never by number.
 *
 * OUTPUT: real <pageset> XML with folio markers  ->  run xml_to_wylie.py on them.
 * RUN: reload the tibetan.works tab first (clears stray loops), then paste in the
 *      console. Chrome may require typing `allow pasting` before the first paste.
 * ========================================================================== */
await (async () => {
  console.log('=== tibetan.works XML fetch on', location.href, '===');
  if (!location.hostname.includes('tibetan.works')) { console.error('Run this on a https://tibetan.works/ tab (same-origin fetch needed).'); return; }

  // ===== FILLED PER REQUEST =====
  const COLLECTION = 'kangyur';          // 'kangyur' (Lhasa H) | 'tengyur' (Derge)
  const targets = [
    // { toh: 543, sanskrit: 'mulakalpa', tib: 'འཇམ་དཔལ་གྱི་རྩ་བའི' },  // tib = distinctive title substring
  ];
  const START = 1, END = 1110, delayMs = 120;   // narrow START/END to go faster
  // ==============================

  const sleep = ms => new Promise(r => setTimeout(r, ms));
  const PUNC = ['་', '།', '༎', '༏', '༐', '༑'];
  const norm = s => { s = (s || '').replace(/\s+/g, ''); for (const p of PUNC) s = s.split(p).join(''); return s; };
  const TIB = /[ༀ-࿿]/;
  const deEnt = s => s.replace(/&#39;/g, "'").replace(/&amp;/g, '&').replace(/&#(\d+);/g, (m, n) => String.fromCharCode(+n));
  const tgt = targets.map(t => ({ ...t, hook: norm(t.tib), done: false }));
  const got = [];

  async function peek(url) {                       // read only enough to see <info>
    const res = await fetch(url, { credentials: 'include' });
    if (!res.ok || !res.body) return '';
    const rd = res.body.getReader(), dec = new TextDecoder();
    let h = '';
    while (h.length < 6000) { const { done, value } = await rd.read(); if (done) break; h += dec.decode(value, { stream: true }); if (h.includes('</info>')) break; }
    try { await rd.cancel(); } catch (e) {}
    const titles = [...h.matchAll(/<title>([\s\S]*?)<\/title>/g)].map(m => m[1].trim());
    return titles.find(t => TIB.test(t)) || '';
  }

  for (let i = START; i <= END && tgt.some(t => !t.done); i++) {
    try {
      const url = 'https://tibetan.works/etext/reader.php?collection=' + COLLECTION + '&index=' + i;
      const title = await peek(url);
      if (i % 25 === 0) console.log('...scanned ' + i);
      const match = title && tgt.find(t => !t.done && norm(title).includes(t.hook));
      if (!match) { await sleep(delayMs); continue; }

      const html = await (await fetch(url, { credentials: 'include' })).text();
      let s = html.indexOf('<?xml-stylesheet'); if (s < 0) s = html.indexOf('<text>');
      const e = html.lastIndexOf('</text>');
      if (s < 0 || e < 0) { console.warn('H' + i + ': no XML block'); await sleep(delayMs); continue; }
      const xml = html.slice(s, e + 7);
      const fm = html.match(/value=['"]([^'"]+\.xml)['"]/i);
      const base = fm ? deEnt(fm[1]).split('/').pop() : ('index' + i + '.xml');
      const fname = (COLLECTION === 'kangyur' ? 'H' : 'toh') + (match.toh || i) + '-' + (match.sanskrit || 'text') + '__' + base;

      const a = Object.assign(document.createElement('a'), { href: URL.createObjectURL(new Blob([xml], { type: 'application/xml' })), download: fname });
      document.body.appendChild(a); a.click(); a.remove(); URL.revokeObjectURL(a.href);
      match.done = true;
      got.push({ toh: match.toh, index: i, title: title.slice(0, 36), file: fname });
      console.log('>>> ' + (match.toh ? 'Toh ' + match.toh : '') + ' @ index ' + i + '  ' + title.slice(0, 36) + '  -> ' + fname);
    } catch (err) { console.warn('index ' + i + ' ERROR: ' + err.message); }
    await sleep(delayMs);
  }
  console.log('=== DONE ==='); console.table(got);
  const miss = tgt.filter(t => !t.done).map(t => 'Toh ' + t.toh + ' (' + t.sanskrit + ')');
  if (miss.length) console.warn('NOT FOUND (widen START/END or fix tib hook):', miss);
  return got;
})();
