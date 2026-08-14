# Data

This directory contains the curated data tables used in the manuscript.

## Files

### `internal_modeling_curated_992.csv`

Curated internal modeling cohort containing 992 peroxidase-like nanozyme
kinetics records.

The deterministic filtering sequence was:

1. Start with 1210 source records.
2. Retain 1002 records labeled as peroxidase-like.
3. Retain 992 records with a positive, unit-convertible apparent Km value.

The target is stored as `target_log10_km`, corresponding to:

`log10(Kmapp / M)`

### `internal_shared_schema_915.csv`

Subset of the internal modeling cohort containing 915 records from the
TMB–H2O2 reaction system.

This table is used for analyses restricted to descriptors available in both
the internal and external corpora.

### `external_validation_curated_322.csv`

Author-curated prospective external-validation corpus containing 322
substrate-specific records from 127 publications published during 2024–2026.

The table retains:

- material formula;
- substrate-specific apparent Km;
- DOI and bibliographic provenance;
- source location;
- curation source;
- curation notes and interpretation information.

DOI, title, journal, year, source location, and other bibliographic fields are
used for provenance and independence auditing. They are not model features.

### `internal_curation_filter_log.csv`

Machine-readable record of the deterministic filtering steps used to construct
the 992-record internal modeling cohort.

### `checksums.sha256`

SHA-256 checksums for verifying that the released CSV files have not changed.

## Units

Apparent Km values are converted to molar units before modeling.

The prediction target is:

`log10(Kmapp / M)`

No target value is imputed.

## Internal-data provenance

The internal source table was derived from the DiZyme 2.0 resource.
The released analysis code contains the deterministic filtering,
unit-conversion, formula-standardization, and target-construction procedures.

Users should consult the original DiZyme publication and resource terms when
reusing the derived internal data.

## External-data provenance

The external dataset was curated specifically for this study.

Eligible entries were primary reports of peroxidase-like nanozyme kinetics
with:

- an explicit positive apparent Km for TMB or H2O2;
- an assignable material formula or composition label;
- sufficient publication provenance;
- an identifiable source location.

All eligible material–substrate pairs were retained from multi-material
publications. No best-performing-material selection rule was applied.

## Important interpretation note

The reported apparent Km is assay- and protocol-dependent. It should not be
interpreted as an intrinsic material property independent of experimental
conditions.
