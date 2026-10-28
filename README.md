# NAMRC 2027 — NASA CNC milling data pipeline (v0.3)

Companion data service for the public research platform: https://www.ariesbank.com/trietphan/publication/namrc/2027/

## What it does
- GitHub Actions retrieves the public historical NASA Milling ZIP from NASA's S3 archive.
- SciPy attempts to extract MATLAB experiment structures and computes descriptive sensor summaries.
- If successful, GitHub Actions commits `public/api/manifest.json` and `public/api/experiments.json`.
- Vercel serves the JSON with CORS access for `https://www.ariesbank.com`.
- No Python or NASA ZIP download is required on the AriesBank web server.

**Research status:** The actual NASA archive could not be downloaded/tested in the build environment. The parser is a starting implementation, not a verified data extraction. It deliberately fails rather than publishing fabricated records. Experiment features, wear labels, grouping, leakage-free validation, and trained prediction are NOT yet implemented. Data may be too large for convenient client loading; inspect JSON size before public use.

## Setup
1. Upload the repository contents to `adonlead/namrc-2027-cnc` (preserving `.github/workflows/nasa-data.yml`).
2. In GitHub, open **Actions → Refresh NASA Milling dataset → Run workflow**.
3. Read the logs. If parsing fails, inspect the `mat_inventory`/archive structure and adjust `scripts/process_nasa.py`; do not assume results are valid.
4. After `public/api/manifest.json` and `public/api/experiments.json` exist, import the repository into Vercel as a **static site** (Framework Preset: Other; Output Directory: `public`; no build command).
5. Open `https://YOUR-PROJECT.vercel.app/api/manifest.json` and verify it contains the expected SHA-256 and experiment count.
6. Replace `VERCEL_PROJECT_URL` in `ariesbank/nasa-data-client.js` with your actual Vercel hostname, then upload the client file to your AriesBank NAMRC directory. It is a data connector only, not a full replacement for the existing page.
7. Update the AriesBank page to call `loadNasaExperiments()` and render returned experimental records; remove the old PHP-relay buttons after integration.

## Notes
- This is historical data, not live CNC telemetry.
- Monthly GitHub Actions checks are for source reproducibility, not new live observations.
- NASA dataset use should be credited according to original dataset documentation.
- Vercel free-plan limits and GitHub Actions quotas are subject to provider terms.
- The NASA source endpoint may change, and GitHub Actions permissions may need enabling under repository settings.
