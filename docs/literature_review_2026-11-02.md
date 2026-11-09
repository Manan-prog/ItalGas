# Literature review: 2D pipe localisation and continuity from vehicle-towed array GPR (IDS Stream UP, Savona)

Conclusion: the published state of the art for our problem is a two-stage pipeline (per-channel hyperbola detection → 3D association/line fitting with explicit uncertainty), run on top of a classical preprocessing + migration chain; no peer-reviewed paper was found that fuses 200+600 MHz specifically for utilities, nor one that reports plan accuracy of a GNSS/IMU-georeferenced vehicle-towed array in urban canyons, so those two parts must be established by us.

Verification key: **[V]** bibliographic details verified via Crossref/DOI/publisher during this session; **[P]** partially verified (title/venue confirmed, content from abstract or secondary summary); **[M]** from memory, not verified here; **[VENDOR]** vendor document.

---

## 1. Classical pipeline (dewow, background removal, time-zero, velocity, migration, time-slices)

- **Cassidy 2009** [V], ch. 5 in Jol (ed.) *GPR Theory and Applications*, Elsevier, DOI 10.1016/B978-0-444-53348-7.00005-3. Use: canonical processing order (dewow/DC removal → time-zero → background removal → gain → filtering → migration) and the warning that gain must be applied *after* background removal and is incompatible with later amplitude attributes.
- **Yelf 2004** [P], "Where is true time zero?", Proc. 10th Int. Conf. GPR (Delft), IEEE; follow-up **Strange & Yelf 2012** [V], 14th ICGPR, DOI 10.1109/icgpr.2012.6254905. Use: time-zero should be set from the ground-surface reflection on the early wavelet, not the first break, and varies with antenna elevation; recommends measuring it per channel against a known-depth target rather than assuming.
- **Özdemir, Demirci, Yiğit, Yilmaz 2014** [V], *Math. Probl. Eng.* 2014:280738, DOI 10.1155/2014/280738 (open access). Use: side-by-side comparison of hyperbolic summation, Kirchhoff, phase-shift, f-k (Stolt) and back-projection on simulated and measured B-scans; good basis for our baseline migration choice. Originals: **Stolt 1978** [V] *Geophysics* DOI 10.1190/1.1440826; **Schneider 1978** [V] *Geophysics* DOI 10.1190/1.1440828.
- **Grasmueck, Weger, Horstmeyer 2005** [V], "Full-resolution 3D GPR imaging", *Geophysics* 70(1) K12–K19, DOI 10.1190/1.1852780. Use: full-resolution 3D migration needs unaliased sampling (≈λ/4) in *both* horizontal directions; our 4.5 cm along-track is fine but the 33.5 cm column spacing is aliased for 600 MHz, so true 3D migration across columns is not justified—use 2D (per-channel) migration plus cross-column association instead.
- **Trinks et al. 2010** [P], *Archaeological Prospection* 17:175–186 (MALÅ MIRA 16-ch 400 MHz, 8 cm spacing); **Linford, Linford, Martin, Payne 2010** [V], *Arch. Prosp.* 17(3), DOI 10.1002/arp.382 (3D-Radar stepped-frequency array). Use: how time-slices are built from array data (per-channel processing, binning to a grid, amplitude envelope, slice thickness ≈ one wavelength).
- **Goodman & Piro 2013** [V], *GPR Remote Sensing in Archaeology*, Springer, DOI 10.1007/978-3-642-31857-3. Use: time-slice construction (binning, inverse-distance interpolation, overlay analysis) and GPS-navigated gridding—the method behind GPR-SLICE.
- **Simi, Bracciali, Manacorda 2008** [V], "Hough transform based automatic pipe detection for array GPR: algorithm development and on-site tests", *IEEE Radar Conf.*, DOI 10.1109/RADAR.2008.4720763. IDS authors; closest published description of the processing philosophy behind the Stream family (vendor-affiliated but peer-reviewed conference).
- **Sharma et al. 2017** [P], "Critical analysis of background subtraction techniques on real GPR data", *Defence Science Journal*, https://publications.drdo.gov.in/ojs/index.php/dsj/article/view/10048. Use: mean, median, SVD, PCA, ICA compared on real data; SVD fails when target and clutter subspaces overlap (relevant for pipes running parallel to travel, which look like clutter).
- Velocity from hyperbola fitting: **Sham & Lai 2016** [V] *NDT&E Int.* DOI 10.1016/j.ndteint.2016.05.002 (fit model including pipe radius and Tx–Rx separation); **Ristić, Petrovački, Govedarica 2009** [V] *Comput. Geosci.* DOI 10.1016/j.cageo.2009.01.003 (joint radius + velocity); **Giannakis, Zhou, Warren, Giannopoulos 2022** [V] *IEEE GRSL* DOI 10.1109/LGRS.2022.3195947 (negative result: velocity and radius are not jointly identifiable from a common-offset hyperbola; fix radius from the pipe class and estimate velocity only); **He & Lai 2023/2024** [P] "Angle-corrected GPR hyperbolic fitting models" SSRN 10.2139/ssrn.4656255, journal version in *TUST* 2024 (oblique crossing flattens the hyperbola and biases velocity upward—critical for us because many pipes run obliquely to the street).
- Vendor: IDS Stream UP / Stream DP product pages [VENDOR] (https://idsgeoradar.com/products/ground-penetrating-radar/stream-up); GPR-SLICE manual [VENDOR]. Useful only for stated hardware facts (dual-polarised HH/VV channels: HH favours transverse, VV longitudinal targets).

## 2. Automatic hyperbola detection

Classical:
- **Capineri, Grande, Temple 1998** [V], *Int. J. Imaging Syst. Technol.* 9(1):51–59, DOI 10.1002/(SICI)1098-1098(1998)9:1<51::AID-IMA7>3.0.CO;2-Q. Hough transform for hyperbolas; first real-time pipe detector.
- **Windsor, Capineri, Falorni 2014** [V], *IEEE GRSL*, DOI 10.1109/LGRS.2013.2248119. Data-pair-labelled generalised Hough: votes from point pairs, reduces parameter space; velocity comes out as a by-product.
- **Pasolli, Melgani, Donelli 2009** [V], *IEEE TGRS*, DOI 10.1109/TGRS.2009.2012701. Pattern-recognition (GA + SVM) classification of detected objects.
- **Maas & Schmalzl 2013** [V], *Comput. Geosci.* 58:116–125, DOI 10.1016/j.cageo.2013.04.012. Viola–Jones (OpenCV Haar cascade) detector + simple Hough for apex/velocity; cheap, trainable on few hundred examples.
- **Mertens, Persico, Matera, Lambot 2016** [V], *IEEE TGRS* 54(1):580–596, DOI 10.1109/TGRS.2015.2462727. Canny edges + analytic hyperbola fit, no velocity prior; reported false-alarm ≤20 %, miss ≤28 % on complex field data—a realistic baseline number.
- **Sagnard & Tarel 2016** [V], *J. Geophys. Eng.* 13(4):491, DOI 10.1088/1742-2132/13/4/491. Template matching with simulated hyperbola templates.
- **Dou, Wei, Magee, Cohn 2017** [V], "Real-time hyperbola recognition and fitting in GPR data", *IEEE TGRS* 55(1):51–62, DOI 10.1109/TGRS.2016.2592679 (OA: https://eprints.whiterose.ac.uk/102528/). Threshold → **column-connection clustering (C3)** → ML classifier → orthogonal-distance hyperbola fit; tested on synthetic + real; later reused on Stream C by Karle 2022. Best classical starting point for us.
- **Zhou, Chen, Li 2019** [V], "Probabilistic mixture model for mapping the underground pipes", *ACM TKDD*, DOI 10.1145/3344721. CFEM (classification EM) + angle-limited Hough; input is noisy GPR detections *with GPS*; output pipe positions and directions—explicitly our problem.

Deep learning (B-scan):
- **Pham & Lefèvre 2018** [V], IGARSS, arXiv 1803.08414. Faster R-CNN; gprMax synthetic augmentation; evaluation qualitative only.
- **Lei et al. 2019** [V bib / M content], *Autom. Constr.* 106:102839, DOI 10.1016/j.autcon.2019.102839. Faster R-CNN detection followed by C3 + DBSCAN + orthogonal-distance fitting [M]; accuracy figures could not be retrieved (publisher blocked).
- **Jaufer et al. 2021** [V], IWAGPR, DOI 10.1109/IWAGPR50767.2021.9843151. Faster R-CNN with two annotation strategies (whole hyperbola vs apex region) to cut false positives; gprMax + field data.
- **Liu, Yue, Liu, Spencer, Cui 2023** [V bib], *TUST* 133:104861, DOI 10.1016/j.tust.2022.104861. DL recognition + localisation of pipelines in B-scans; numbers not retrieved.
- **Jafary, Shojaei, Ehinger 2025** [V], *Sensors* 25(20):6414, DOI 10.3390/s25206414. YOLOv8/YOLOv11/Mask R-CNN compared on real multi-receiver B-scans; Mask R-CNN keypoint (apex) F1 = 0.822, bbox F1 = 0.867 on held-out test; apices clustered with 3D DBSCAN and fitted with RANSAC lines (RMSE 0.06, units not stated in abstract). Closest published analogue of a per-channel-detect-then-link pipeline with real data and no synthetic augmentation.
- Reviews: **Liu C. et al. 2025** [V], *Discover Applied Sciences*, DOI 10.1007/s42452-025-06791-y; **Gao & Hu 2026** [V], *Sensors* 26(9):2708, DOI 10.3390/s26092708.

Deep learning (3D / array):
- **Yamaguchi, Mizutani, Nagayama 2021** [V], *IEEE TGRS*, DOI 10.1109/TGRS.2020.3030079 (short version: *Seisan Kenkyu* 72(4):329, 2020). 3D-CNN classifies sub-volumes as transverse pipe / longitudinal pipe / none (~91 % accuracy per secondary summary, ~6 % better than 2D-CNN on transverse pipes), then Kirchhoff migration extracts 3D positions. Note it treats longitudinal pipes as a separate class—our key failure mode.
- **Bai et al. 2023** [V], *Applied Sciences* 13:7540, DOI 10.3390/app13137540. Vehicle-mounted 3D GPR; depthwise-separable 3D CNN (66.9 % fewer parameters, similar accuracy); includes survey-line orientation correction for non-orthogonal crossings.
- **Zhou et al. 2024** [V bib], *TUST*, DOI 10.1016/j.tust.2024.105819. 3D U-Net segmentation of pipes in C-scan volumes.
- **Karle, Boldt, Thiele, Thoennessen 2022** [V], *ISPRS Archives* XLIII-B1-2022:85, https://isprs-archives.copernicus.org/articles/XLIII-B1-2022/85/2022/. IDS Stream C (23 usable channels, 4 cm along-track): per-channel Dou-2016 hyperbola detection → 3D point cloud → clustering → line approximation; 7 of 8 reference pipes found; the *diagonal* pipe and a 63 mm pipe were missed; many unrecorded pipes detected (no ground truth to score them). Most directly comparable published study to ours.

Accuracy caveat (all of the above): mAP/F1 figures are on per-paper datasets with different annotation protocols; none reports plan-position error in cm against excavated ground truth except via total-station test sites.

## 3. Linear-feature extraction and probabilistic utility mapping

- **Dell'Acqua, Sarti, Tubaro, Zanzi 2004** [V], "Detection of linear objects in GPR data", *Signal Processing* 84:785–799, DOI 10.1016/j.sigpro.2003.12.010. 3D Radon/Hough on the migrated volume with multiresolution search; the one published Radon-on-volume utility method; memory-heavy.
- **Böniger & Tronicke 2012** [V], "Subsurface utility extraction and characterization: combining GPR symmetry and polarization attributes", *IEEE TGRS* 50(3):736–746, DOI 10.1109/TGRS.2011.2163413 (OA: https://publishup.uni-potsdam.de/opus4-ubp/frontdoor/index/index/docId/36101). Phase-symmetry attribute to enhance pipe-like features + PCA polarisation (linearity, angle) from two broadside polarisations → depolarisation attribute that separates utilities. Directly applicable: we have HH and VV channels.
- **Böniger & Tronicke 2010** [V], *J. Archaeol. Sci.*, DOI 10.1016/j.jas.2009.09.049. Target-specific 3D attributes (energy, similarity, curvature-like).
- **Trinks & Hinterleitner 2020** [V], "Beyond amplitudes: multi-trace coherence analysis for GPR data imaging", *Remote Sensing* 12(10):1583, DOI 10.3390/rs12101583 (OA). Seismic coherence/similarity transferred to dense 3D GPR (8 × 4 cm); states it is directly transferable only when cross-line spacing is dense—our 33.5 cm columns limit cross-column coherence to the 200 MHz data.
- **Zhao, Forte, Pipan 2016** [V], *Pure Appl. Geophys.*, DOI 10.1007/s00024-016-1355-3. GLCM texture attributes on time-slices.
- **Leckebusch, Weibel, Bühler 2007** [V], *Near Surface Geophysics*, DOI 10.3997/1873-0604.2007033. 3D morphological opening/closing on the data cube for semi-automatic linear-feature extraction.
- **Chen & Cohn 2011** [V, PDF read], "Buried utility pipeline mapping based on multiple spatial data sources: a Bayesian data fusion approach", IJCAI-11 pp. 2411–2417, https://www.ijcai.org/Proceedings/11/Papers/402.pdf. State vector per point of interest t = (x, y, θ); Gaussian prior from records, Gaussian likelihoods from GPR hyperbola observations and from a pipe-linearity term; associations by Joint Compatibility Branch and Bound; NNSF Mahalanobis gate at 0.95/0.99 to accept/reject links. This is the template for a probabilistic utility map with per-segment covariance.
- **Dou, Wei, Magee, Cohn 2016** [V], "3D buried utility location using a marching-cross-section algorithm for multi-sensor data fusion", *Sensors* 16(11):1827, DOI 10.3390/s16111827 (OA). Tracks pipes across successive cross-sections (initialise, march, associate, update, manage), handles curved and intersecting pipes; directly maps to marching along our swath with 5 columns as the cross-section.
- **Oguntoye et al. 2024** [V, PDF read], *Sensors* 24(11):3559, DOI 10.3390/s24113559. Each segment gets a Gaussian perpendicular to its line (σ and weight per source); products of distributions across sources give a 2D probability map; prunes below threshold; 2D only. Simple recipe for "fusion of multiple passes".
- **Zhang & Cai 2026** [V bib], *Adv. Eng. Informatics* 68:104093, DOI 10.1016/j.aei.2025.104093. Uncertainty-aware (Bayesian) position/orientation/radius estimation; details not retrieved.
- **Jafary et al. 2025** (above): DBSCAN + RANSAC line fitting for continuity.
- Tensor voting (**Medioni, Lee, Tang 2000** [V] book, DOI 10.1016/B978-044450353-4): no GPR application found—negative search result. Ridge detection: **Lindeberg 1998** [V], *IJCV*, DOI 10.1023/A:1008097225773 (scale-normalised Hessian ridges, the basis of Frangi).

## 4. Multi-frequency fusion (600 + 200 MHz)

Negative result: no peer-reviewed utility-detection paper fusing 200 and 600 MHz was found. Closest:
- **Wang et al. 2024** [V], "An NSCT-based multifrequency GPR data-fusion method for concealed damage detection", *Buildings* 14:2657, DOI 10.3390/buildings14092657. Fuses 600 MHz and 200 MHz radargrams with non-subsampled contourlet transform; road and tunnel voids, not pipes; quality judged by entropy/gradient metrics, not detection rate.
- **Fang et al. 2026** [V], *J. Imaging* 12(1):52, DOI 10.3390/jimaging12010052. Convolutional sparse representation fusion after spatial registration (100/200/400 MHz); registration step is the reusable part since our 600 and 200 rows are 0.347 m apart along-track.
- SPIE 2024 regional-energy fusion [P], DOI 10.1117/12.3015645 (900/1600 MHz).
- *J. Appl. Geophys.* 2017 dual-frequency subpavement-drain paper [P], DOI 10.1016/j.jappgeo.2017.09.019 (authors not captured).
- **IDS RIS MF Hi-Mod** page [VENDOR] advertises automatic 200+600 fusion; no method or evidence given.
Recommendation: fuse at the *detection* level (independent hyperbola/line detections per frequency, combined in the probabilistic map with frequency-specific σ), not at the image level, since image fusion papers never evaluate localisation.

## 5. Positioning, georeferencing, achievable accuracy, standards

- **Gabryś & Ortyl 2020** [V, PDF read], "Georeferencing of multi-channel GPR—accuracy and efficiency of mapping of underground utility networks", *Remote Sensing* 12(18):2945, DOI 10.3390/rs12182945 (OA). Leica/IDS Stream C; GNSS RTN vs GNSS with 1-PPS time synchronisation vs total station; two-run repeatability against a surveyed metal tape/grille gives RMSE ±5.2 cm with PPS; without PPS discrepancies up to ±20 cm at segment ends, typically ±10 cm; a drainage grille lay 25–40 cm from its mapped record. Confirms that time synchronisation between GNSS and radar trigger is the first-order term (relevant: our `.index` holds µs timestamps).
- **Šarlah, Podobnikar, Ambrožič, Mušič 2020** [V], *Remote Sensing* 12(8):1228, DOI 10.3390/rs12081228 (OA). Kinematic GPR + robotic total station; <15 cm in urban environment; mean spatial error 30 cm and 40 cm at two real sites; cites 100–163 cm errors in earlier urban studies.
- **Barzaghi, Cazzaniga, Pagliari, Pinto 2016** [V], "Vision-based georeferencing of GPR in urban areas", *Sensors* 16(1):132, DOI 10.3390/s16010132 (OA, Italian group). Photogrammetric trajectory recovery where GNSS degrades; few-cm accuracy; relevant as a fallback for our >1 m epochs.
- **Böniger & Tronicke 2010** [V], *IEEE TGRS* 48(10):3792–3798, DOI 10.1109/TGRS.2010.2048332. Determines positioning latency by comparing forward and reverse profiles over a known target with three fidelity measures—directly reusable to calibrate our GPS lever arm/latency from reciprocal passes.
- **Grasmueck & Viggiano 2007** [V], *IEEE TGRS* 45(1):130–137, DOI 10.1109/TGRS.2006.882253. Laser positioning at 20 Hz fused in real time; sets the cm-level reference.
- **Metje, Hojjati, Beck, Rogers 2020** [V], *Proc. ICE – Municipal Engineer*, DOI 10.1680/jmuen.18.00055. Four firms on a 15 × 20 m car park vs trial pits: ~70 % of assets detected; QL-B1 vertical accuracy "rarely achieved"; two techniques raised confidence but not accuracy.
- **PAS 128:2022** (BSI) [standard; figures from secondary sources, not the BSI text]: QL-B1 ±150 mm horizontal or ±15 % of depth; QL-B2 ±250 mm or ±40 %; QL-B3 ±500 mm horizontal only; QL-A ±50 mm. 2022 revision reportedly requires two detection techniques for all B levels (from draft summary, unverified). **ASCE 38-22** [P]: quality levels describe provenance, not numeric accuracy.
- NovAtel Inertial Explorer documentation [VENDOR]: lever-arm/boresight refinement, DMI aiding, tightly coupled mode—confirms our odometer in `.index` can be used as a DMI constraint.
Negative result: no paper reports plan accuracy of a GNSS/IMU vehicle-towed array in urban canyons; our own repeat-pass and surface-feature checks must fill this.

## 6. Cross-disciplinary analogues (with the assumption to check)

- Seismic coherence: **Bahorich & Farmer 1995** [V] *TLE* DOI 10.1190/1.1437077; **Marfurt et al. 1998** [V] *Geophysics* DOI 10.1190/1.1444415 (semblance); **Gersztenkorn & Marfurt 1999** [V] DOI 10.1190/1.1444651 (eigenstructure). Assumption: traces are regularly and densely sampled in both directions with consistent wavelet—true along-track, false across our 5 columns and across the two frequency rows.
- Ant tracking: **Pedersen, Randen, Sønneland, Steen 2002** [P], SEG Expanded Abstracts / EAGE G037. Assumption: features are discontinuities (faults) in a continuous reflector; pipes are *bright continuous ridges*, so apply to a ridge/energy attribute, not to a discontinuity attribute.
- **Frangi et al. 1998** [V], MICCAI, LNCS 1496:130–137, DOI 10.1007/BFb0056195. Hessian vesselness. Assumption: tubular feature has a Gaussian cross-section at a known scale range and is brighter than background; in migrated time-slices the pipe cross-section is set by the migration aperture and the 33.5 cm column spacing, so scales must be anisotropic.
- **Tupin et al. 1998** [V], *IEEE TGRS* 36(2):434–453, DOI 10.1109/36.662728. Local ratio line detector + MRF global linking for roads in speckled SAR. Assumption: multiplicative speckle noise; GPR clutter is additive/structured, so the ratio detector needs re-derivation, but the MRF linking step transfers as is.
- **Clark, Peek, Putman 2014** [V], *ApJ* 789:82, DOI 10.1088/0004-637X/789/1/82. Rolling Hough Transform for linear filaments in HI radio maps—a local, windowed Hough giving per-pixel orientation distributions. Assumption: features are long relative to the window and isotropically sampled; check against our anisotropic grid.
- **Offringa et al. 2010** [V], *MNRAS*, DOI 10.1111/j.1365-2966.2010.16471.x (SumThreshold) and **2012** [V] *A&A* DOI 10.1051/0004-6361/201118497 (SIR operator). Multi-scale line-shaped thresholding and scale-invariant dilation along lines. Assumption: lines are axis-parallel in time–frequency; pipes are not, so apply after Radon/Hough rotation.
- **Stoppa et al. 2024** [V], *A&A*, DOI 10.1051/0004-6361/202451663. U-Net + probabilistic Hough for satellite trails—a template for "learned mask then geometric line fit".

## 7. Open-source tools

- **gprMax** (Warren, Giannopoulos, Giannakis 2016 [V], *Comput. Phys. Commun.* 209:163–170, DOI 10.1016/j.cpc.2016.08.020): FDTD simulator; arbitrary Tx/Rx placement so cross-antenna pairs and both frequency rows can be simulated; used by Pham 2018 and Jaufer 2021 for training data.
- **GPRPy** (Plattner 2020 [V], *The Leading Edge* 39(5):332, DOI 10.1190/tle39050332.1): single-channel; reads Sensors&Software, GSSI, MALÅ; no IDS, no arrays.
- **readgssi** (Nesbitt, PyPI/GitHub [P]): GSSI only; multichannel support partial.
- **RAGU** (Tober & Christoffersen, PyPI 0.1.x [P], GPL-3): picking GUI for ice sounders and pulseEKKO; not array-aware; not peer-reviewed.
- **ImpDAR** (Lilien et al. 2020 [V], *Ann. Glaciol.*, DOI 10.1017/aog.2020.44): Python; Stolt/phase-shift/Kirchhoff migration, GSSI/pulseEKKO readers; single channel.
- **RGPR** (Huber & Hans 2018 [V], ICGPR, DOI 10.1109/ICGPR.2018.8441658): R package, single channel.
- **MultichannelGPR** (Wunderlich 2021 [V], *ArchéoSciences* 45(1), DOI 10.4000/archeosciences.10100): MATLAB, the only open tool written for multichannel arrays (MALÅ MIRA etc.); useful as a reference implementation of per-channel equalisation and binning.
- **segyio** (Equinor, GitHub [P], not peer-reviewed): SEG-Y I/O only; useful if we export per-channel volumes for seismic-attribute tools.
- Commercial/vendor: IDS GRED HD 3D, GPR-SLICE [VENDOR].
Bottom line: nothing reads IDS `.data/.index/.path`; keep our own reader module.

---

## Recommended pipeline for our setup (synthesis)

1. Calibrate before detecting: per-channel time-zero from the ground-surface wavelet (Yelf 2004), GNSS–radar latency/lever arm from forward–reverse passes over a fixed surface target (Böniger & Tronicke 2010), and map trajectory σ per scan from the post-processed track (Gabryś & Ortyl 2020 show time sync dominates).
2. Preprocess per channel: DC/dewow → time-zero → moving-median background removal (keep SVD out: it suppresses along-street pipes; Sharma 2017) → gain only for display/detection images, never for attributes (Cassidy 2009).
3. Velocity: fit hyperbolas with fixed nominal radius per pipe class and oblique-angle correction (Giannakis 2022; He & Lai 2024); log velocity per street with its spread; report as derived.
4. Baseline detector: C3 clustering + orthogonal-distance hyperbola fit (Dou 2017), run on every one of the 29 channels independently; score false-alarm/miss against the Mertens 2016 numbers (≤20 %/≤28 %) as the first benchmark.
5. Per-channel 2D migration (Kirchhoff or Stolt, Özdemir 2014) along-track only; do not 3D-migrate across columns (Grasmueck 2005: 33.5 cm is aliased at 600 MHz).
6. Attributes on the migrated volume: Hessian/Frangi ridge (anisotropic scales) and phase-symmetry + HH/VV depolarisation (Böniger & Tronicke 2012); coherence/semblance across columns only for the 200 MHz row (Trinks & Hinterleitner 2020).
7. Continuity: treat each scan's five columns as a cross-section and run a marching-cross-section tracker (Dou 2016) with JCBB association and a Mahalanobis gate (Chen & Cohn 2011); fall back to DBSCAN + RANSAC line fits (Jafary 2025) as the simple baseline; try Rolling Hough (Clark 2014) on time-slices for orientation priors.
8. Two frequencies: detect independently in 600 and 200 MHz rows, register along-track (0.347 m offset), and fuse at the detection level in the probability map with frequency-specific σ; image-level NSCT/CSR fusion (Wang 2024; Fang 2026) only as an experiment, scored on localisation not entropy.
9. Probabilistic map: per-segment Gaussian perpendicular to the line whose σ is the quadrature sum of trajectory σ, hyperbola-apex σ and lateral column-spacing term (Oguntoye 2024 recipe; Chen & Cohn 2011 posterior); multiple passes multiply distributions.
10. Output quality tags against PAS 128 QL-B1/B2/B3 thresholds (±150/±250/±500 mm) so every segment carries a standard-referenced accuracy class; label provisional until ItalGas GIS and surface features arrive.
11. Learned detector only after the baseline: Mask R-CNN keypoint-style apex detection on real B-scans (Jafary 2025), trained per held-out street/day, with gprMax synthetic data (Pham 2018) as augmentation; adopt only if it beats step 4 on the same held-out streets.
12. Validate without GIS: repeat-pass consistency (Gabryś & Ortyl RMSE method), crossing-swath consistency at intersections, surface features (manholes, valves) as anchors, and the physical constraint that a pipe's detections must lie on one velocity-consistent hyperboloid across HH and VV channels.
13. Expected published ceiling for plan accuracy: ≈5 cm repeatability with good time-sync and GNSS (Gabryś & Ortyl 2020), 15–40 cm mean error in real urban sites (Šarlah 2020); set our target and uncertainty reporting accordingly.
14. Record every borrowed method in REFERENCES.md with the "assumption to check" from section 6.

---

## Reference table

| Author | Year | Venue | URL / DOI | What we use |
|---|---|---|---|---|
| Cassidy N. | 2009 | Ch. 5, Jol (ed.) *GPR Theory and Applications*, Elsevier | 10.1016/B978-0-444-53348-7.00005-3 | Processing order; gain after background removal |
| Yelf R. | 2004 | Proc. 10th Int. Conf. GPR, IEEE [P] | (IEEE Xplore, Proc. GPR 2004) | Time-zero from ground-surface wavelet |
| Strange A., Yelf R. | 2012 | 14th ICGPR | 10.1109/icgpr.2012.6254905 | True time range of a GPR system |
| Özdemir C., Demirci Ş., Yiğit E., Yilmaz B. | 2014 | *Math. Probl. Eng.* | 10.1155/2014/280738 | Migration method comparison for baseline |
| Stolt R. | 1978 | *Geophysics* | 10.1190/1.1440826 | f-k migration original |
| Schneider W. | 1978 | *Geophysics* | 10.1190/1.1440828 | Kirchhoff migration original |
| Grasmueck M., Weger R., Horstmeyer H. | 2005 | *Geophysics* 70(1) | 10.1190/1.1852780 | λ/4 sampling criterion; no 3D migration across our columns |
| Trinks I. et al. | 2010 | *Archaeological Prospection* 17:175–186 [P] | (Wiley; DOI not verified) | Array time-slice workflow (MIRA) |
| Linford N., Linford P., Martin L., Payne A. | 2010 | *Archaeological Prospection* 17(3) | 10.1002/arp.382 | Array time-slice workflow (3D-Radar) |
| Goodman D., Piro S. | 2013 | Springer book | 10.1007/978-3-642-31857-3 | Time-slice construction, GPS gridding |
| Leckebusch J., Weibel A., Bühler F. | 2007 | *Near Surface Geophysics* | 10.3997/1873-0604.2007033 | 3D morphological feature extraction |
| Wunderlich T. | 2021 | *ArchéoSciences* 45(1) | 10.4000/archeosciences.10100 | Open multichannel array processing reference (MATLAB) |
| Simi A., Bracciali S., Manacorda G. | 2008 | IEEE Radar Conf. (IDS authors) | 10.1109/RADAR.2008.4720763 | Hough pipe detection for array GPR, on-site tests |
| Sharma P. et al. | 2017 | *Defence Science Journal* [P] | https://publications.drdo.gov.in/ojs/index.php/dsj/article/view/10048 | Background-removal comparison; SVD caveat |
| Sham J., Lai W. | 2016 | *NDT&E International* | 10.1016/j.ndteint.2016.05.002 | Hyperbola model with radius and Tx–Rx offset |
| Ristić A., Petrovački D., Govedarica M. | 2009 | *Computers & Geosciences* | 10.1016/j.cageo.2009.01.003 | Joint radius–velocity estimate |
| Giannakis I., Zhou F., Warren C., Giannopoulos A. | 2022 | *IEEE GRSL* | 10.1109/LGRS.2022.3195947 | Velocity/radius non-identifiability |
| He W., Lai W. | 2023 (preprint) / 2024 TUST [P] | SSRN / *TUST* | 10.2139/ssrn.4656255 | Oblique-angle hyperbola correction |
| Capineri L., Grande P., Temple J. | 1998 | *Int. J. Imaging Syst. Technol.* | 10.1002/(SICI)1098-1098(1998)9:1<51::AID-IMA7>3.0.CO;2-Q | Hough for hyperbolas |
| Windsor C., Capineri L., Falorni P. | 2014 | *IEEE GRSL* | 10.1109/LGRS.2013.2248119 | Pair-labelled generalised Hough |
| Pasolli E., Melgani F., Donelli M. | 2009 | *IEEE TGRS* | 10.1109/TGRS.2009.2012701 | Pattern-recognition classification |
| Maas C., Schmalzl J. | 2013 | *Computers & Geosciences* 58 | 10.1016/j.cageo.2013.04.012 | Viola–Jones cascade + Hough |
| Mertens L., Persico R., Matera L., Lambot S. | 2016 | *IEEE TGRS* 54(1) | 10.1109/TGRS.2015.2462727 | Prior-free detector; FA ≤20 %, miss ≤28 % benchmark |
| Sagnard F., Tarel J.-P. | 2016 | *J. Geophys. Eng.* 13(4) | 10.1088/1742-2132/13/4/491 | Template matching |
| Dou Q., Wei L., Magee D., Cohn A. | 2017 | *IEEE TGRS* 55(1) | 10.1109/TGRS.2016.2592679; https://eprints.whiterose.ac.uk/102528/ | C3 clustering + ODR fit (baseline detector) |
| Zhou X., Chen H., Li J. | 2019 | *ACM TKDD* | 10.1145/3344721 | CFEM mixture + angle-limited Hough with GPS |
| Pham M.-T., Lefèvre S. | 2018 | IGARSS | arXiv 1803.08414 | Faster R-CNN + gprMax augmentation |
| Lei W. et al. | 2019 | *Automation in Construction* 106 | 10.1016/j.autcon.2019.102839 | Faster R-CNN + fitting [content M] |
| Jaufer R. et al. | 2021 | IWAGPR | 10.1109/IWAGPR50767.2021.9843151 | Annotation strategy to cut false positives |
| Liu H. et al. | 2023 | *TUST* 133 | 10.1016/j.tust.2022.104861 | DL pipeline localisation in B-scans [bib only] |
| Jafary P., Shojaei D., Ehinger K. | 2025 | *Sensors* 25(20) | 10.3390/s25206414 | Keypoint F1 0.822; DBSCAN+RANSAC 3D lines |
| Yamaguchi T., Mizutani T., Nagayama T. | 2021 | *IEEE TGRS* | 10.1109/TGRS.2020.3030079 | 3D-CNN transverse/longitudinal classes + Kirchhoff |
| Bai X. et al. | 2023 | *Applied Sciences* 13 | 10.3390/app13137540 | Lightweight 3D CNN on vehicle array; orientation correction |
| Zhou Y. et al. | 2024 | *TUST* | 10.1016/j.tust.2024.105819 | 3D U-Net on C-scan volume [bib only] |
| Karle N., Boldt M., Thiele A., Thoennessen U. | 2022 | *ISPRS Archives* XLIII-B1-2022 | https://isprs-archives.copernicus.org/articles/XLIII-B1-2022/85/2022/ | Stream C per-channel detect → cluster → lines; 7/8 pipes; diagonal missed |
| Liu C. et al. | 2025 | *Discover Applied Sciences* | 10.1007/s42452-025-06791-y | Review of B-scan processing/detection |
| Gao S., Hu D. | 2026 | *Sensors* 26(9) | 10.3390/s26092708 | Recent utility-GPR review |
| Dell'Acqua A., Sarti A., Tubaro S., Zanzi L. | 2004 | *Signal Processing* 84 | 10.1016/j.sigpro.2003.12.010 | 3D Radon/Hough for linear objects |
| Böniger U., Tronicke J. | 2012 | *IEEE TGRS* 50(3) | 10.1109/TGRS.2011.2163413 | Phase symmetry + HH/VV depolarisation attribute |
| Böniger U., Tronicke J. | 2010 | *J. Archaeol. Sci.* | 10.1016/j.jas.2009.09.049 | Target-specific 3D attributes |
| Trinks I., Hinterleitner A. | 2020 | *Remote Sensing* 12(10) | 10.3390/rs12101583 | Coherence on dense 3D GPR; spacing caveat |
| Zhao W., Forte E., Pipan M. | 2016 | *Pure Appl. Geophys.* | 10.1007/s00024-016-1355-3 | Texture attributes on slices |
| Chen H., Cohn A. | 2011 | IJCAI-11 pp. 2411–2417 | https://www.ijcai.org/Proceedings/11/Papers/402.pdf | Bayesian (x,y,θ) fusion, JCBB, Mahalanobis gate |
| Dou Q., Wei L., Magee D., Cohn A. et al. | 2016 | *Sensors* 16(11) | 10.3390/s16111827 | Marching-cross-section tracker |
| Oguntoye K. et al. | 2024 | *Sensors* 24(11) | 10.3390/s24113559 | Per-segment Gaussian probability map, multi-source |
| Zhang Y., Cai H. | 2026 | *Adv. Eng. Informatics* 68 | 10.1016/j.aei.2025.104093 | Uncertainty-aware pipe pose [bib only] |
| Medioni G., Lee M., Tang C. | 2000 | Elsevier book | 10.1016/B978-044450353-4 | Tensor voting (no GPR use found) |
| Lindeberg T. | 1998 | *IJCV* | 10.1023/A:1008097225773 | Scale-selected ridge detection |
| Wang J. et al. | 2024 | *Buildings* 14 | 10.3390/buildings14092657 | NSCT fusion of 600+200 MHz (damage, not pipes) |
| Fang L. et al. | 2026 | *J. Imaging* 12(1) | 10.3390/jimaging12010052 | CSR fusion; multi-frequency registration |
| SPIE regional-energy fusion [P] | 2024 | SPIE Proc. 12973 | 10.1117/12.3015645 | Fusion threshold selection (900/1600 MHz) |
| JAG dual-frequency drain paper [P] | 2017 | *J. Appl. Geophys.* 146 | 10.1016/j.jappgeo.2017.09.019 | Dual-frequency configuration evidence |
| IDS RIS MF Hi-Mod page | n.d. | Vendor [VENDOR] | https://idsgeoradar.com/products/ground-penetrating-radar/ris-mf-hi-mod | Claim of automatic 200+600 fusion, no method |
| IDS Stream UP / DP pages | 2020– | Vendor [VENDOR] | https://idsgeoradar.com/products/ground-penetrating-radar/stream-up | HH/VV orientation sensitivity; hardware facts |
| Gabryś M., Ortyl Ł. | 2020 | *Remote Sensing* 12(18) | 10.3390/rs12182945 | ±5.2 cm RMSE with PPS sync; ±10–20 cm without |
| Šarlah N., Podobnikar T., Ambrožič T., Mušič B. | 2020 | *Remote Sensing* 12(8) | 10.3390/rs12081228 | <15 cm urban (TPS); 30–40 cm mean site error |
| Barzaghi R., Cazzaniga N., Pagliari D., Pinto L. | 2016 | *Sensors* 16(1) | 10.3390/s16010132 | Vision georeferencing fallback in GNSS-poor streets |
| Böniger U., Tronicke J. | 2010 | *IEEE TGRS* 48(10) | 10.1109/TGRS.2010.2048332 | Latency calibration by forward/reverse passes |
| Grasmueck M., Viggiano D. | 2007 | *IEEE TGRS* 45(1) | 10.1109/TGRS.2006.882253 | cm-level positioning reference |
| Metje N., Hojjati A., Beck A., Rogers C. | 2020 | *Proc. ICE – Municipal Engineer* | 10.1680/jmuen.18.00055 | PAS 128 field trial: ~70 % detection, QL-B1 vertical rarely met |
| BSI PAS 128:2022 | 2022 | Standard (figures via secondary sources) | https://gogeomatics.ca/magazine/between-the-poles/assessment-of-the-pas-128-quality-standard-for-underground-utility-location | QL-B1/B2/B3/A tolerances |
| ASCE 38-22 | 2022 | Standard [P] | https://blog.emlid.com/asce-38-quality-levels/ (secondary) | Quality levels = provenance, not accuracy |
| NovAtel Inertial Explorer docs | n.d. | Vendor [VENDOR] | https://novatel.com/products/waypoint-software/inertial-explorer | Lever arm/boresight, DMI aiding, tightly coupled |
| Bahorich M., Farmer S. | 1995 | *The Leading Edge* 14 | 10.1190/1.1437077 | Coherence cube |
| Marfurt K., Farmer S., Kirlin R., Bahorich M. | 1998 | *Geophysics* 63 | 10.1190/1.1444415 | Semblance coherence |
| Gersztenkorn A., Marfurt K. | 1999 | *Geophysics* | 10.1190/1.1444651 | Eigenstructure coherence |
| Pedersen S., Randen T., Sønneland L., Steen Ø. | 2002 | SEG Expanded Abstracts / EAGE G037 [P] | (no DOI found) | Ant tracking |
| Frangi A., Niessen W., Vincken K., Viergever M. | 1998 | MICCAI, LNCS 1496 | 10.1007/BFb0056195 | Hessian vesselness |
| Tupin F. et al. | 1998 | *IEEE TGRS* 36(2) | 10.1109/36.662728 | Line detector + MRF linking |
| Clark S., Peek J., Putman M. | 2014 | *ApJ* 789:82 | 10.1088/0004-637X/789/1/82 | Rolling Hough Transform |
| Offringa A. et al. | 2010 | *MNRAS* | 10.1111/j.1365-2966.2010.16471.x | SumThreshold line-shaped flagging |
| Offringa A., van de Gronde J., Roerdink J. | 2012 | *A&A* | 10.1051/0004-6361/201118497 | Scale-invariant rank operator |
| Stoppa F. et al. | 2024 | *A&A* | 10.1051/0004-6361/202451663 | U-Net + probabilistic Hough for trails |
| Warren C., Giannopoulos A., Giannakis I. | 2016 | *Comput. Phys. Commun.* 209 | 10.1016/j.cpc.2016.08.020 | gprMax simulation of our array |
| Plattner A. | 2020 | *The Leading Edge* 39(5) | 10.1190/tle39050332.1 | GPRPy (single channel) |
| Nesbitt I. | n.d. | PyPI/GitHub [P] | https://readgssi.readthedocs.io | readgssi (GSSI only) |
| Tober B., Christoffersen M. | n.d. | PyPI [P] | https://pypi.org/project/ragu | RAGU picking GUI |
| Lilien D. et al. | 2020 | *Annals of Glaciology* | 10.1017/aog.2020.44 | ImpDAR migration code |
| Huber E., Hans G. | 2018 | 17th ICGPR | 10.1109/ICGPR.2018.8441658 | RGPR (R) |
| Equinor | n.d. | GitHub [P] | https://github.com/equinor/segyio | SEG-Y I/O |

Not verified / not found: a peer-reviewed 200+600 MHz utility-fusion paper; a vehicle-towed-array GNSS/IMU urban plan-accuracy study; an IDS/Manacorda journal paper describing Stream UP processing (only the 2008 radar-conference paper exists); tensor voting applied to GPR. Lei 2019, Liu 2023, Zhou 2024 and Zhang & Cai 2026 are confirmed bibliographically but their accuracy figures could not be retrieved (publisher/API blocks).