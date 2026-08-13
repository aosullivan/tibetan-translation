# Wylie/EWTS source download — research findings and method recommendation

This document records where each missing-Wylie text for the *Dispelling Error* quotation catalogue can be obtained online, in EWTS / Wylie or Tibetan Unicode (with a conversion path). It also recommends a concrete download approach.

## Context

The existing `other-sources/*.wylie.txt` files use a very specific format:

```
# --- metadata (from <info>) ---
# organization=acip collection=k_lha_sa volume=39 index=6
# title: ...
# section: ...
# volume: ca
# status: ...
# --- body ---

### page index=1 id=222A
//rgya gar skad du/ ...
```

The `organization=acip collection=k_lha_sa` metadata shows these were generated from **ACIP Lhasa Kanjur** texts, with a small wrapper script adding the YAML-like header. ACIP's transliteration is EWTS-style (`AR+Ya`, `D+haR+ma`, `na ma`, etc.). Any new texts ideally want to land in the same format so they slot in cleanly alongside `Ākāśagarbha.wylie.txt`, `upali.EWTS.txt`, `Saṃcayagāthā.wylie.txt`, etc.

---

## The four practical sources

### 1. Asian Legacy Library (ALL) — successor of ACIP

The website is at https://asianlegacylibrary.org/library/. Under **DATA DOWNLOADS** it offers three public-domain ZIPs:

| Collection | URL | Size |
|---|---|---|
| Kangyur | https://all-library-docs.s3.us-west-2.amazonaws.com/KANGYUR.zip | 128 MB |
| Tengyur | https://all-library-docs.s3.us-west-2.amazonaws.com/TENGYUR.zip | 254 MB |
| Sungbum | https://all-library-docs.s3.us-west-2.amazonaws.com/SUNGBUM.zip | 270 MB |

These archives are the same source that produced our existing `other-sources/` files — they are ACIP EWTS plain-text files in the Lhasa-Kanjur / Derge-Tengyur traditions. License is public-domain (CC0).

**Pros:** Single archive ⇒ no per-text fetching. Matches the existing format almost directly (only the metadata-wrapper script needs to be re-applied). Has both the Kangyur sūtras *and* the Lhasa-Kanjur tantra volumes.

**Cons:** The complete Sungbum is 270 MB; we'd only need a handful of texts from each. Hand-extracting the relevant texts after unzipping is required.

### 2. Christian Steinert's ACIP archive

The archive of ACIP Release 6 is at https://acip-archive.christian-steinert.de/. It exposes individual text files at predictable URLs:

- ACIP-original: `https://acip-archive.christian-steinert.de/flat/TD####X_T.TXT`
- Auto-converted Wylie: `https://acip-archive.christian-steinert.de/wylie/TD####X_T.TXT`

where `T` = Tengyur, `K` = Kangyur, `D|L|X` = Derge/Lhasa/etc. edition. Confirmed working: e.g. `TD3871M_T.TXT` (Bodhicaryāvatāra), `TD4024N_T.TXT` (Uttaratantra), `TD4090I1_T.TXT` (Abhidharmakośa autocommentary I).

The full Wylie bundle is also offered as a single ~90 MB zip:
- `https://resources.christian-steinert.de/download/acip-release6-wylie.zip`

**Pros:** Per-text fetching by Tohoku catalogue number is easy to script.

**Cons:** Christian Steinert's Wylie was **machine-converted** from ACIP transliteration and "has only been checked in a few cases" — the original ACIP version is the more reliable input. The site only lists Mahāyāna sūtras and śāstras by English name; tantras are not as well surfaced in the English by-title catalogue (though they are in the ACIP file system, e.g. KL0494M = *Vairocanābhisaṃbodhi*).

### 3. rKTs (Resources for Kanjur & Tanjur Studies, University of Vienna)

http://www.rkts.org/etexts/repository.php gives full access to every major canonical edition: Derge, Cone, Lhasa, Phugbrag, Lithang, London, Narthang, Peking, Stog, plus four Tengyur editions and nine Old Tantra collections.

The e-texts are now **BDRC-OCR derived** (March 2025) — Unicode Tibetan, not Wylie. They are explicitly raw OCR with no manual correction.

**Pros:** Total coverage. Has all tantras (which ACIP lacks). Multiple editions for cross-checking. Includes Old Tantra collections (e.g. for *Guhyagarbha*, which is not in standard Kangyurs).

**Cons:** OCR — error-prone. Unicode Tibetan, so a Wylie conversion (via `pyewts`) is needed. Per-text downloads via the website navigator rather than bulk URLs.

### 4. Esukhia / OpenPecha GitHub repositories

- `Esukhia/derge-kangyur` (archived, moved to `OpenPecha-Data/P000001`)
- Companion `derge-tengyur` repo
- Released ZIPs: `deka_vol_txt_v1905.zip` (volumes as plain text), `UT4CZ5369-200106.zip` (TEI)

These are **proofread** (Esukhia–Barom 2014–2018) Derge Kangyur in **Tibetan Unicode**. Higher quality than OCR; lower coverage than ACIP+rKTs combined.

**Pros:** Proofread, normalised, well-documented format.

**Cons:** Unicode, not Wylie; conversion needed. Tengyur volume is comparatively less mature than the Kangyur volume.

---

## Per-text plan

Mapping each missing-Wylie text to the best source. The columns indicate **primary** (recommended) and **fallback** options.

| Text (Toh) | Primary | Fallback | Notes |
|---|---|---|---|
| Bodhicaryāvatāra (D3871) | ACIP Tengyur ZIP → `TD3871M_T.TXT` | rKTs Derge | Direct URL works: `acip-archive…/wylie/TD3871M_T.TXT` |
| Mahāyāna-Uttaratantra-śāstra (D4024) | ACIP Tengyur ZIP → `TD4024N_T.TXT` | rKTs | Confirmed downloadable |
| Mūlamadhyamakakārikā (D3824) | ACIP Tengyur ZIP | rKTs | TD3824 in Tengyur dbu ma |
| Ratnāvalī (D4158) | ACIP Tengyur ZIP | rKTs | |
| Pramāṇavārttika (D4210) | ACIP Tengyur ZIP → `TD4210N_T.TXT` | rKTs | Confirmed in catalogue |
| Suhṛllekha (D4182) | ACIP Tengyur ZIP | rKTs | |
| Abhidharmakośa (D4090) | ACIP Tengyur ZIP → `TD4090I1_T.TXT`, `TD4090I2_T.TXT` | rKTs | autocommentary in two parts |
| Caryāmelāpakapradīpa (D1803) | ACIP Tengyur ZIP | rKTs | Tantra commentaries section of Tengyur |
| Cittaviśuddhiprakaraṇa (D1804) | ACIP Tengyur ZIP | rKTs | |
| Śrī-Hevajrapañjikā Muktāvalī (D1189) | ACIP Tengyur ZIP | rKTs | |
| Sahajāloka commentary on Kṛṣṇayamāri (≈D1924 or nearby) | rKTs Derge Tengyur | ACIP archive | Find Tohoku number first |
| Vajraśekhara (D480) | rKTs Derge Kangyur (tantra) | ALL Kangyur ZIP | Not in ACIP English catalogue; in Kangyur rgyud-collection |
| Vajrapañjara / Ḍākinī-vajrapañjara (D419) | rKTs Derge Kangyur | ALL Kangyur ZIP | |
| Hevajra-tantra (D417, D418) | rKTs Derge Kangyur | ALL Kangyur ZIP | |
| Guhyasamāja (D442) | rKTs Derge Kangyur | ALL Kangyur ZIP | |
| Kṛṣṇayamāri-tantra (D469) | rKTs Derge Kangyur | ALL Kangyur ZIP | |
| Cakrasaṃvara (D368) | rKTs Derge Kangyur | ALL Kangyur ZIP | |
| Vairocanābhisaṃbodhi-tantra (D494) | ALL Kangyur ZIP → `KL0494M_T.TXT` | ACIP archive | Confirmed in ACIP Kangyur listing |
| Paramādya / Paramādi-buddha (D487, D488) | rKTs Derge Kangyur | ALL Kangyur ZIP | |
| Saṃpuṭa-tantra (D381) | rKTs Derge Kangyur | ALL Kangyur ZIP | |
| Jñānatilaka (D422) | rKTs Derge Kangyur | ALL Kangyur ZIP | |
| Vimalaprabhā Kālacakra commentary (D1347) | ACIP Tengyur ZIP | rKTs Derge Tengyur | Long text; in rgyud-commentary section |
| Saddharmasmṛtyupasthāna-sūtra (D287) | ALL Kangyur ZIP | rKTs | mdo-sde large text |
| Guhyagarbha-tantra (NGB; not in main Kangyur) | rKTs Old Tantra → "Gdg Derge rnying rgyud" or "Gpb Paltsek rnying rgyud" | — | Outside the standard Kangyur — use Nyingma rgyud collection |
| Samayapañca (Padmavajra, Tengyur) | ACIP Tengyur ZIP | rKTs | Find Tohoku number |
| Tattvajñānasiddhi | rKTs Derge Tengyur | — | Tohoku catalogue lookup needed |
| Indrabhūti, *Summary of the Intent of Empowerment* | rKTs Derge Tengyur | — | |
| Vajragīti (*rdo rje glu*) | rKTs Derge Kangyur (rgyud) | — | |
| *dbang yon tan rim pa* | rKTs Derge Tengyur (Kālacakra section) | — | |
| Vidyottama (*rig pa mchog*) | rKTs Old Tantra | — | Ancient Mahāyoga, NGB |
| Guhyakoṣa-tantra (*gsang ba'i mdzod*) | rKTs Old Tantra | — | Title not in standard Kangyur; search NGB |

---

## Recommended workflow

The most efficient path balances completeness against post-processing work:

**Step 1 — Bulk download Asian Legacy Library archives.** This is the single biggest move and covers the majority of citations in one operation:

```sh
curl -O https://all-library-docs.s3.us-west-2.amazonaws.com/KANGYUR.zip
curl -O https://all-library-docs.s3.us-west-2.amazonaws.com/TENGYUR.zip
unzip KANGYUR.zip -d kangyur/
unzip TENGYUR.zip -d tengyur/
```

After unzipping, every text we need from the Kangyur, the Tengyur śāstra side, AND the Lhasa-Kanjur tantras will be available locally. (Sungbum is not needed for chapters 1–2 of *Dispelling Error*.)

**Step 2 — Build a small extraction script.** A short Python (or shell) script that:

1. Maps each Tohoku number / catalogue ID (e.g. `KL0494M`, `TD3871M`) to the corresponding file in the unzipped archive.
2. Re-wraps the body with the project's `# organization=acip collection=… # title: … # --- body ---` metadata header so the new files match the existing convention.
3. Writes to `other-sources/<sanitized-title>.wylie.txt`.

The existing `Ākāśagarbha.wylie.txt`, etc., can serve as exact templates. Information for the header (title, section, volume, page IDs) can be lifted from the ACIP source file's own headers, which are simple and consistent.

**Step 3 — For the gap texts not in ACIP (Guhyagarbha, Vidyottama, Sahajāloka, Indrabhūti, *dbang yon tan rim pa*, *Vajragīti*, etc.):** download from rKTs in Unicode Tibetan and convert with `pyewts`:

```sh
pip install pyewts
python3 - <<'PY'
import pyewts
converter = pyewts.pyewts()
with open('input-tibetan.txt') as f:
    print(converter.toWylie(f.read()))
PY
```

Note the rKTs Unicode is OCR-derived and "raw" — for any quotation that's important to verify, cross-check the rKTs Wylie against the Wylie *as Drakpa Gyaltsen quotes it* in our own `source/NN-*.txt` files. Discrepancies are likely just OCR slips.

**Step 4 — Treat the Christian Steinert ACIP archive as a third-tier fallback.** Per-text URL fetching is convenient when a single passage needs sanity-checking, but the bulk approach in step 1 is faster overall.

---

## Effort estimate

Roughly:

- ACIP bulk download + unzip: ~10 minutes (network-bound), ~400 MB on disk.
- Mapping script + first batch of ~10 high-priority texts (Bodhicaryāvatāra, Uttaratantra, Mūlamadhyamakakārikā, Ratnāvalī, Pramāṇavārttika, Suhṛllekha, Abhidharmakośa, Caryāmelāpakapradīpa, Cittaviśuddhiprakaraṇa, Hevajrapañjikā Muktāvalī, Vairocanābhisaṃbodhi, Saddharmasmṛtyupasthāna): ~1–2 hours of scripting.
- rKTs Unicode-Tibetan downloads + pyewts conversion + manual review for ~6 tantra-corpus texts: ~2–3 hours.
- NGB / Nyingma rgyud lookups for *Guhyagarbha*, *Vidyottama*, etc.: an additional ~1–2 hours since these need catalogue search rather than direct Toh-number lookup.

So a single day's work would get the project to roughly 90% coverage of the cited sources for chapters 1 and 2 in the same EWTS Wylie format already in use.

---

## Open questions for the user

1. Do you want **all** cited works downloaded as full texts, or only the immediate surrounding passages of each cited quotation? The former is simpler to script; the latter saves disk space and reading effort.
2. For the tantras that the ACIP Lhasa Kanjur doesn't cover well — is the Derge Kanjur edition (rKTs, OCR-derived) acceptable, or do you need a hand-input edition?
3. Some quotations are from texts that may simply not exist in any digital corpus we can reach (e.g. *Sahajāloka*, *dbang yon tan rim pa*) — do you want me to mark those as "physical edition only" or try harder to find them in obscure repositories?
