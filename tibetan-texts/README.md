# Tibetan source texts

Organised by **practice cycle**, then by **genre**. `tibetan_translations/` mirrors this
structure exactly — an English translation lives in the folder matching its source.

## Filename conventions

| Suffix / prefix | Meaning |
|---|---|
| `*.wylie.txt` | EWTS / Wylie transliteration |
| `*.bo.txt`, `*.tibetan.txt` | Tibetan Unicode |
| `*.xml` | ACIP-style Tibetan Unicode XML — input to `xml_to_wylie.py` |
| `*.clean.*` | OCR output after cleanup |
| `toh####-` | Tohoku catalogue number (Derge Tengyur) |
| `H###` | Lhasa Kangyur number |
| `W…`, `UT…`, `VE…`, `UTMW…` | BDRC work / etext identifiers |

---

## `vajrayogini/` — Vajrayoginī, Nāro Khecarī, Vajravārāhī

| File | What it is |
|---|---|
| `naro-khacho/` | BDRC IE1PD5284, *sa skya'i mkha' spyod be'u bum* — 69 etext volumes of the Sakya Nāro Khecarī collection. Includes `MANIFEST.txt`, the collection `karchag.wylie.txt`, and `khachoma-v2/` (per-text Wylie extracts with their own README) |
| `phagmo/` | Pawo Tsuglag Trengwa, *phag mo rnam bshad* — commentary on Nāropa's Vajrayoginī secret sādhana, *zab mo rnam 'byed*. Split into `phagmo.txt` and `phagmo-part2.txt` |
| `khyentse-vajrayogini.txt` | Jamyang Khyentse Wangpo, *…nā ro mkha' spyod kyi byin rlabs bya tshul mthong ba don ldan* — the blessing ritual |
| `konchog-lhundrup.txt` | Konchog Lhundrub, *bskyed rim rnal 'byor bcu gcig gi rnam par bshad pa mkha' spyod lam bzang gsal ba'i nyin byed* — the eleven yogas of the generation stage |
| `naro-khachoma-sadhana-tsarpa-ocr.{wylie,tibetan}.txt` | Tsarpa-tradition Nāro Khecarī sādhana, OCR'd |
| `phakmo-sadhana.txt` | Vajrayoginī generation-stage daily practice (*rgyun khyer*) |

## `kalachakra/` — Kālacakra

| File | What it is |
|---|---|
| `bka'-'gyur_H0371(rgyud.ka003).{wylie.txt,xml}` | *Laghukālacakratantra* — the condensed root tantra, Lhasa Kangyur 371 |
| `kalacakra-root-H370…H375*.wylie.txt` | Five shorter root-tantra texts: *Sekoddeśa* (Toh 361), *rgyud phyi ma snying po* (Toh 363), *bsdus pa'i rgyud rgyal po'i snying po*, *rgyud kyi snying po* (Toh 364), *dbang gi rab tu byed pa* |
| `bstan-'gyur_toh1347.01(rgyud.tha002).xml`, `…1347.02(rgyud.da001).xml` | *Vimalaprabhā* — Puṇḍarīka's great Kālacakra commentary, Derge Tengyur vols *tha* and *da* |
| `vimalaprabha-complete.wylie.txt` | Wylie of both volumes — **use this one** |
| `vimalaprabha.wylie.txt`, `vimalaprabha-da.wylie.txt` | Earlier partial passes (vol 1 only, vol 2 only) |
| `vimalaprabhā-1…5-*.wylie.txt` | The same commentary split by its five chapters: *lokadhātu*, *adhyātma*, *abhiṣeka*, *sādhana*, *jñāna* |

## `hevajra-lamdre/` — Hevajra, Sampuṭa, Lamdre

| File | What it is |
|---|---|
| `lamdre-vol1.txt` | *lam 'bras* vol 1 — dkar chag plus the Virūpa and Indian-lineage histories |
| `lamdre-vol13.txt`, `sonam-chokdrup.txt` | Further *lam 'bras* volumes, Dzongsar Khamje edition |
| `toh1186-vajragupta-hevajra-panjika.wylie.txt` | *rdo rje'i tshig gi snying po bsdus pa'i dka' 'grel* — Hevajra pañjikā |
| `toh1198.wylie.txt`, `bstan-'gyur_toh1198(rgyud.cha001).xml` | Abhayākaragupta, *Āmnāyamañjarī* — commentary on the *Sampuṭa* explanatory tantra |

## `vows-and-samaya/` — the three vows, root downfalls, guru devotion

| File | What it is |
|---|---|
| `trisaṃvara-nirdeśa-parivarta.{wylie.txt,xml}` | *sdom pa gsum bstan pa'i le'u* — the Three Vows chapter of the *Ratnakūṭa* |
| `sdom-gsum.txt` | Shākya Chokden, gsung 'bum volume on the three vows |
| `gorampa-sdom-gsum.txt` | Gorampa Sonam Sengge, collected works — three-vows volume |
| `drakpa-gyaltsen-20verses.txt` | Drakpa Gyaltsen's commentary on Candragomin's *Twenty Verses on the Bodhisattva Vows* |
| `drakpa-gyaltsen-50vs.txt`, `bla-ma-dang.txt` | *bla ma lnga bcu dang rtsa ltung bcu bzhi'i rnam bshad* — Fifty Verses on the Guru + the Fourteen Root Downfalls. **Two extractions of the same work**; `bla-ma-dang.txt` is the fuller one |
| `tsarchen-siddhis.txt` | *bla ma lnga bcu pa'i 'grel pa dngos grub rin po che'i sgo 'byed* — a second Fifty Verses commentary |
| `toh2488-manjusrikirti-vajrayana-mulapatti-tika.wylie.txt` | Mañjuśrīkīrti, *Vajrayāna-mūlāpatti-ṭīkā* |
| `damtshig.txt` | Samayavajra (*dam tshig rdo rje*) — meditation, recitation and fire-offering cycle |
| `upali.wylie.txt` | *Upāliparipṛcchā* — *'phags pa 'dul ba rnam par gtan la dbab pa nye bar 'khor gyis zhus pa* |

## `tantra-general/` — maṇḍala ritual, tantric śāstra

| File | What it is |
|---|---|
| `abhayakara-vajravali.txt` | Abhayākaragupta, *Vajrāvalī* — the maṇḍala-ritual compendium |
| `vajrasekhara.wylie.txt` | *Vajraśekhara* — *gsang ba rnal 'byor chen po'i rgyud rdo rje rtse mo* |
| `tshul-gsum.txt` | Tripiṭakamāla, *Nayatrayapradīpa* — *tshul gsum gyi sgron ma* |
| `tattvajnanasiddhi.{bo,wylie}.txt` | *Tattvajñānasiddhi* — *dpal de kho na nyid ye shes grub pa* |
| `toh2228-kuddalapada-acintyakrama.wylie.txt` | Kuddālapāda, *Acintyakrama-upadeśa* |
| `toh2492-ganacakravidhi.wylie.txt` | Kṛṣṇācārya, *Gaṇacakravidhi* |
| `toh3718…`, `toh3719…` | Jñānākara, *Mantrāvatāra* and its *vṛtti* |

## `sutra-and-shastra/` — sūtra and non-tantric treatise

| File | What it is |
|---|---|
| `ārya-dharmasaṃgīti-nāma.xml` | *Dharmasaṃgīti-sūtra* |
| `lions-roar.txt` | Jamgön Kongtrul, *theg pa chen po rgyud bla ma'i tshig 'grel* — word commentary on the *Uttaratantra* |
| `lions-roar-UTMW29022_….txt` | *phyir mi ldog pa seng ge'i nga ro* — a second witness of the same commentary. Collated in `../tibetan_translations/sutra-and-shastra/lions-roar-collation.md` |
| `toh4038-asanga-viniscayasamgrahani-COMPLETE.wylie.txt` | Asaṅga, *Yogācārabhūmi-Viniścayasaṃgrahaṇī* — Esukhia/OpenPecha Derge text (complete) |
| `toh4038-…-part1.wylie.txt` | Partial ACIP release of the same |
| `toh4090-vasubandhu-abhidharmakosabhasya.wylie.txt` | Vasubandhu, *Abhidharmakośabhāṣya* |

## `biography-and-history/` — namthar, genealogy, catalogues

| File | What it is |
|---|---|
| `zhuchen/` | Zhuchen Tsultrim Rinchen's autobiography — 190 etext files (`VE00EGS1016703_*`) plus `zhuchenautobiographymap.html` |
| `tsarpa-lineage/` | Three Sakya/Tsarpa biographies from BDRC W1KG9245 v2 — Morchen Kunga Lhundrub (b. 1654), Nesarwa Kunga Lekpai Jungne (1704–1760), Dingmawa Labsum Gyaltsen. PDF + Wylie per page range |
| `sakya-dungrabs.txt` | *sa skya gdung rabs* — the Sakya genealogy |
| `sakya-vol6.txt` | *sa skya'i chos 'byung gces bsdus*, vol 6 — Sakya religious history |
| `karchag.{pdf,wylie.txt,tibetan.txt}` | *bod kyi lo rgyus rnam thar phyogs bsgrigs (1–30)* dkar chag — catalogue to the BDRC "Historical Biographies of Tibet" set |

## `bodong/` — Bodong Panchen Chogle Namgyal (1375–1451)

| File | What it is |
|---|---|
| `bodong-namthar.txt` | 'Jigs med 'bangs, *bo dong paN chen gyi rnam thar ngo mtshar gyi dga' ston* (BDRC W28838) |
| `bodong-bzo-rig-v2-pp215-263.pdf` | *mkhas pa 'jug pa'i bzo rig* — the craft-science section, pp. 215–263 |
| `bodong-bzo-rig-painting-pp252-263.pdf` | The painting sub-section, pp. 252–263 |
| `bodong-bzo-rig-painting.MACHINE-OCR-DRAFT.wylie.txt` | Machine OCR draft — **not a proofread source text** |
| `bodong-painting-3model-ocr.txt` | Three-engine OCR comparison of the same pages |

## `collected-works/` — gsung 'bum volumes not yet broken out

| File | What it is |
|---|---|
| `khyentse-vol.txt` | Jamyang Khyentse Wangpo, collected works |
| `UTMW29307_….txt` | Sakya collected works vol 25 (*ra*) — dkar chag, Mahākāla and samaya material |

## `misc/`

| File | What it is |
|---|---|
| `sman-bla-bdun-mched.txt` | *sangs rgyas sman bla mched bdun gyi rjes gnang* — Medicine Buddha seven-brothers permission rite |
| `W23702-v50_img527-528.*` | Two folios of auspiciousness verses (*bkra shis pa'i tshigs su bcad pa*) |

## `_unidentified/` — OCR'd scans not yet matched to a catalogued work

These were extracted from page ranges without recording which work they belong to.
The OCR is degraded in places; see the `highlights` files in the mirrored
translations folder for the passages that came through cleanly.

| File | Notes |
|---|---|
| `W20578-v67_img366-374.*` | 9 folios, apparently a tantric or meditation manual |
| `W22642-v6_img265-282.*` | 18 folios; large stretches corrupt |
| `W8751-v7_img222-241.*` | 20 folios |
| `W23681-v5_img153-177.*`, `W23681-sadhana.wylie.txt` | Tibetan Unicode OCR. The `-sadhana` extract begins at folio 159, at a chapter opening (*bam po bcu pa'i le'u bcu gsum pa*) |

---

## Known duplicates

Three byte-identical pairs were found and the redundant copy moved to `../_to_delete/`:

| Removed | Identical to |
|---|---|
| `manjusrikirti.txt` | `vows-and-samaya/toh2488-manjusrikirti-vajrayana-mulapatti-tika.wylie.txt` |
| `Historical Biographies of Tibet.txt` | `biography-and-history/karchag.wylie.txt` |
| `vimalaprabha.xml` | `kalachakra/bstan-'gyur_toh1347.01(rgyud.tha002).xml` |

`drakpa-gyaltsen-50vs.txt` and `bla-ma-dang.txt` are the same work but *not* identical
files, so both were kept — see the note under `vows-and-samaya/`.
