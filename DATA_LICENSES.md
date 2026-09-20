# Data attribution and licensing

The MIT license applies to this project's original software and documentation. It does not replace the licenses of third-party data or software. Derived data tables and research figures are shared under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), with the source attributions below. The unpublished manuscript is not included.

## KU Leuven

Tiukhova, Elena; Van Landuyt, Dimitri; Snoeck, Monique. *Open Data, Private Learners: A De-Identified Dataset for Learning Analytics Research*, version v1. https://doi.org/10.5281/zenodo.17087849

The source record's API identifies the license as `cc-by-4.0`. Also cite its descriptor: Tiukhova, E., Van Landuyt, D., Baesens, B. & Snoeck, M. *Open data, private learners: a de-identified student activity and performance dataset for learning analytics*. Scientific Data (2026). https://doi.org/10.1038/s41597-026-06821-3

Changes: roster deduplication; original-course-page joins; cumulative activity-window aggregation; train/calibration/test partitions; derived features, predictions, bootstrap intervals and aggregate metrics. Source pseudonyms are retained in the optional row-level package for reproducibility; these records are not newly collected data.

## Open University Learning Analytics Dataset

Kuzilek, J., Hlosta, M. & Zdrahal, Z. *Open University Learning Analytics dataset*. Scientific Data 4, 170171 (2017). https://doi.org/10.1038/sdata.2017.171

Data record: https://doi.org/10.6084/m9.figshare.5081998.v1 — CC BY 4.0, as identified by the Figshare API.

Changes: day-28 eligibility filtering; activity aggregation over days 0–27; non-success endpoint construction; person-disjoint source/target partitions; derived predictions and person-clustered interval estimates.

Original raw datasets are not redistributed here or in the companion Drive package. Obtain them from the cited providers. Existing input-code hash files record the earlier analytical runs; they do not certify the current packaging or documentation code. Git commits identify the public code version; the data-package manifest identifies exact distributed data files.
