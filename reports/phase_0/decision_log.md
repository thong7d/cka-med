# P0 — Decision log

| ID | Decision | Evidence | Status |
|---|---|---|---|
| D1 | Use SLAKE English and bundled KG as the primary benchmark | Dataset card reports CC BY 4.0; metadata audit has train/validation/test and KG | Approved for P0 |
| D2 | Use Qwen2.5-VL-3B-Instruct as default backbone | Public and ungated model API/config probe; Qwen Research License applies | Approved for P0 |
| D3 | Keep MedGemma optional | Its files require authenticated HAI-DEF access; live probe returned 401 GatedRepo | Not a dependency |
| D4 | Do not use images or models at P0 beyond access probes | No P0 implementation or GPU training exists | Required |
| D5 | Treat CXR metadata filters as candidates, not confirmed CXR views | Audit explicitly uses modality X-Ray and Chest/Lung location | Approved |
| D6 | Use CXR only as a slice after image audit | No images/pixel hashes/downloaded yet | Exploratory |
| D7 | No new clinical gold labels | Scope contract and plan prohibit synthetic diagnosis/absence/premise labels | Approved |
| D8 | Keep the final PoC to one candidate and strong baselines | Plan and survey show overlap with prior work; novelty is not pre-validated | Approved |
