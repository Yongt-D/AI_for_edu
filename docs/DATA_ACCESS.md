# Data access

## Original datasets

- KU Leuven: https://doi.org/10.5281/zenodo.17087849 — download `dataset.zip` and `course_info.json`.
- OULAD: https://doi.org/10.6084/m9.figshare.5081998.v1 — `scripts/download_oulad.py` uses the provider's Figshare API.

See [data attribution and licenses](../DATA_LICENSES.md) for citations, CC BY 4.0 terms and the changes made by this analysis.

## Companion Google Drive package

**Status: package prepared; upload and public download remain to be verified.**

Designated folder: [Google Drive](https://drive.google.com/drive/folders/1JM6_PfNNw40j4FWhqAvTGxRweyQwsyAr?usp=sharing). Look for `AI_for_edu_data_v1.0.0.zip`; the folder link alone does not establish that the package has been uploaded. The GitHub repository contains code and aggregate results. Until the companion package is published, use the full refitting workflow to regenerate individual-level files.

- Filename: `AI_for_edu_data_v1.0.0.zip`
- Size: 18,795,926 bytes
- SHA-256: `cd5d801c19ca00cbfb49d5089bae16642d722363896021273a4cc3713d90389e`
- Machine-readable download status: [data_download.json](data_download.json)
- Per-file checksums: [DATA_MANIFEST.json](DATA_MANIFEST.json)

The package contains 22 derived data files, including features, prediction rows and partition identities. It excludes raw datasets and the manuscript. Source pseudonyms support reproducible linkage and must not be treated as permission to identify individuals.

## Verify and extract

After downloading, compare the checksum before extraction:

```powershell
Get-FileHash AI_for_edu_data_v1.0.0.zip -Algorithm SHA256
Expand-Archive -LiteralPath AI_for_edu_data_v1.0.0.zip -DestinationPath .
```

Run extraction from the repository root so `data/processed_v2/` and `outputs/revision_v2/`, `outputs/revision_v3/` retain their relative paths. On macOS/Linux, use `shasum -a 256` or `sha256sum`, then `unzip`. Compare individual extracted files against DATA_MANIFEST.json if needed. Follow [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for numerical validation and figure regeneration.
