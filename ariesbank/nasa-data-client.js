// Replace VERCEL_PROJECT_URL with the actual *.vercel.app production domain.
// Import with <script src="nasa-data-client.js" defer></script>.
const NAMRC_API_BASE = 'https://VERCEL_PROJECT_URL';
async function loadNasaExperiments() {
  const [manifestResponse, dataResponse] = await Promise.all([
    fetch(`${NAMRC_API_BASE}/api/manifest.json`, {cache:'no-store'}),
    fetch(`${NAMRC_API_BASE}/api/experiments.json`, {cache:'no-store'})
  ]);
  if (!manifestResponse.ok || !dataResponse.ok) throw new Error(`Data API unavailable: ${manifestResponse.status}/${dataResponse.status}`);
  const manifest=await manifestResponse.json();
  const dataset=await dataResponse.json();
  if (!Array.isArray(dataset.records) || !dataset.records.length) throw new Error('No experimental records in NASA data response');
  return {manifest,records:dataset.records};
}
