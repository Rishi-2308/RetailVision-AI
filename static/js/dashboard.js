// dashboard.js - loads numbers from /api/dashboard and draws Chart.js charts
const charts = {};
const PALETTE = ['#4f8cff', '#22c1c3', '#f7b731', '#eb5757', '#9b59b6', '#2ecc71', '#e67e22'];

function draw(id, type, labels, datasets, opts) {
  if (charts[id]) charts[id].destroy();
  charts[id] = new Chart(document.getElementById(id), {
    type: type, data: { labels: labels, datasets: datasets },
    options: Object.assign({ responsive: true, plugins: { legend: { display: type !== 'bar' || datasets.length > 1 } } }, opts || {})
  });
}

function fill(data) {
  const set = (k, v) => { document.getElementById('c-' + k).textContent = v; };
  set('total_shoppers', data.total_shoppers);
  set('avg_dwell', data.avg_dwell);
  set('avg_engagement', data.avg_engagement);
  set('high_engagement', data.high_engagement);
  set('most_visited_zone', data.most_visited_zone);
  set('most_common_behavior', data.most_common_behavior);
  document.getElementById('empty').classList.toggle('d-none', data.total_shoppers > 0);

  const bl = Object.keys(data.behavior_distribution);
  draw('ch-behavior', 'doughnut', bl, [{ data: bl.map(k => data.behavior_distribution[k]), backgroundColor: PALETTE }]);
  const el = Object.keys(data.engagement_distribution);
  draw('ch-level', 'bar', el, [{ label: 'Shoppers', data: el.map(k => data.engagement_distribution[k]), backgroundColor: '#4f8cff' }]);

  const zn = data.zones.map(z => z.zone);
  draw('ch-dwell', 'bar', zn, [{ label: 'Avg dwell (s)', data: data.zones.map(z => +(z.avg_dwell || 0).toFixed(1)), backgroundColor: '#22c1c3' }]);
  draw('ch-count', 'bar', zn, [{ label: 'Shoppers', data: data.zones.map(z => z.shoppers), backgroundColor: '#f7b731' }]);
  draw('ch-zoneeng', 'bar', zn, [{ label: 'Avg engagement', data: data.zones.map(z => +(z.avg_eng || 0).toFixed(1)), backgroundColor: '#9b59b6' }],
       { scales: { y: { min: 0, max: 100 } } });

  const behaviors = [...new Set(data.behavior_by_zone.map(r => r.behavior))];
  const zones = [...new Set(data.behavior_by_zone.map(r => r.zone))].sort();
  draw('ch-bz', 'bar', zones, behaviors.map((b, i) => ({
    label: b, backgroundColor: PALETTE[i % PALETTE.length],
    data: zones.map(z => { const r = data.behavior_by_zone.find(x => x.zone === z && x.behavior === b); return r ? r.n : 0; })
  })), { scales: { x: { stacked: true }, y: { stacked: true } } });
}

async function load() {
  const v = document.getElementById('video-filter').value;
  try {
    const res = await fetch(window.DASH_API + (v ? '?video_id=' + v : ''));
    fill(await res.json());
  } catch (e) { console.error('Dashboard load failed', e); }
}
document.getElementById('video-filter').addEventListener('change', load);
load();
