# Data access

## Original datasets

- KU Leuven: https://doi.org/10.5281/zenodo.17087849 — download `dataset.zip` and `course_info.json`.
- OULAD: https://doi.org/10.6084/m9.figshare.5081998.v1 — `scripts/download_oulad.py` uses the provider's Figshare API.

See [data attribution and licenses](../DATA_LICENSES.md) for citations, CC BY 4.0 terms and the changes made by this analysis.

## Companion Google Drive package

**Status: publicly available. Anonymous ZIP download and SHA-256 verification passed on 20 September 2026.**

Download [AI_for_edu_data_v1.0.0.zip](https://drive.google.com/uc?export=download&id=1HoSkyFWVetEhL8B19NV-seETIfqMLDoa), or open the [Google Drive folder](https://drive.google.com/drive/folders/1JM6_PfNNw40j4FWhqAvTGxRweyQwsyAr?usp=sharing) for the ZIP, checksum and manifest. The GitHub repository contains code and aggregate results.

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
