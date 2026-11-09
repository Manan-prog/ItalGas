# Ground-truth search, second round (3 Nov 2026)

Agent report, verified only as far as stated. Downloads and derived files are in the session scratchpad
(not synced): scratchpad/dl_gt/, scratchpad/derived_gt/, scripts in scratchpad/scripts_gt/.

## New source: Comune di Savona PUC sheet DF S.2.1 "Reti di urbanizzazione" (May 2011)
- 1:10,000 vector PDF with separate layers for gas distribution, water distribution, foul/storm sewer, HV/MV power lines. No depth, no attributes.
- URL: https://municipium-images-production.s3-eu-west-1.amazonaws.com/s3/6691/allegati/pianificazione-territoriale/puc/piano-urbanistico-comunale/reti_di_urbanizzazione_-df_s_2_1-_df_s_2_2.pdf (listed at https://www.comune.savona.it/it/documenti_pubblici/descrizione-fondativa-puc).
- Licence: the Comune site states CC BY 4.0 for published documents; the sheet does not state the origin of the network layers, so the right to reuse them is UNCLEAR (CLAUDE.md: ask before use).
- Georeferenced by the agent against the SIT sewer vectors: scale 1.000, NCC 0.709; SIT vertices on PUC sewer pixels median 0.00 m, p90 1.69 m (n = 8,675).
- Coverage: Via Nizza samples every 5 m (n = 1,616): gas line within 5 m of 56 percent, within 10 m of 87 percent; water 66 / 81 percent. Corso Ricci (n = 827): gas 66 / 92 percent; water 61 / 88 percent. Gas line drawn at median 2.45 m (Via Nizza) and 4.03 m (Corso Ricci) from the sewer line.
- Usable for: presence and continuity of a gas line along a street, coarse lateral check (about +-5 m). Not for metre-level accuracy.

## Negative or pending
- ATEM gas tender "Savona 2 Nord-Est" (Comune di Savona contracting authority): not published; ARERA still reviewing values (resolutions 24/2026, 145/2026). The precedent tender ATEM Savona 1 (Provincia, Appalto 2228) supplied DWG network maps per comune (UTM/WGS84, DN, year, pressure) but excludes Savona city and its cloud link returns 404.
- Italgas 2026 works plan: no works in the Comune di Savona.
- Regione Liguria WFS: broadband fibre M2707 (no limitation) has no segment within 15 m of either street; "2i Rete Gas" points none in the bbox; no regional water-network layer.
- OpenStreetMap (ODbL): no gas features within 30 m of the two streets; 2 hydrants (44.2922830, 8.4554748 Via Nizza; 44.3152673, 8.4716655 Corso Ricci), 4 fire-service inlets, 4 water taps, 3 power towers.
- Not found: PUGSS or sottosuolo cadastre, public SINFI viewer, theses with Savona utility ground truth, water network map (operator unresolved: Acque Pubbliche Savonesi vs Iren / Acquedotto di Savona).
- Italgas map requests: MyItalgas "Coordinamento sottoservizi" needs an account; an unverified snippet reports about 4 weeks, DWG/PDF 1:250, maps "indicativa".

## Realistic route to the Italgas map
1. Ask Italgas Reti directly for a GIS export (SHP/GPKG/DWG with CRS) of the surveyed streets: material, DN, pressure class, laying year, survey/accuracy class, valves.
2. Ask Comune di Savona Settore 4 for the PUC network DWGs.
3. The Comune holds the network "stato di consistenza" (DM 226/2011) for the tender; accesso civico possible but may be refused.
4. Otherwise the map is issued to bidders when ATEM Savona 2 is published.
