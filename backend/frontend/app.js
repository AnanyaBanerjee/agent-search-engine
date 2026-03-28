/* ============================================================
   Agent Search Engine — frontend JS
   ============================================================ */

const API = '';   // same-origin; change to 'http://localhost:8000' for local dev


/* ---- Utilities ---- */
function $(id) { return document.getElementById(id); }

const STATUS_COLORS = {
  online:  { bg: 'rgba(78,205,196,.15)',  fg: '#4ecdc4', border: 'rgba(78,205,196,.4)'  },
  offline: { bg: 'rgba(255,128,128,.15)', fg: '#ff8080', border: 'rgba(255,128,128,.4)' },
  stale:   { bg: 'rgba(255,204,0,.15)',   fg: '#ffcc00', border: 'rgba(255,204,0,.4)'   },
  unknown: { bg: 'rgba(136,146,164,.15)', fg: '#8892a4', border: 'rgba(136,146,164,.4)' },
};

function buildAgentCard(r, showScore = true) {
  const c = r.agent_card;
  const tags = [...(c.tags || [])];
  const skills = c.skills || [];
  const health = r.health || {};
  const statusKey = health.status || 'unknown';
  const sc = STATUS_COLORS[statusKey] || STATUS_COLORS.unknown;

  const tagsHtml = tags.length
    ? `<div class="tags">${tags.map(t => `<span class="tag">${t}</span>`).join('')}</div>`
    : '';

  const skillsHtml = skills.length
    ? `<details class="skills">
        <summary>${skills.length} skill${skills.length > 1 ? 's' : ''}</summary>
        ${skills.map(s => `
          <div class="skill-item">
            <div class="skill-name">${s.name}</div>
            <div class="skill-desc">${s.description}</div>
            ${(s.tags || []).length ? `<div class="tags">${s.tags.map(t => `<span class="tag">${t}</span>`).join('')}</div>` : ''}
          </div>
        `).join('')}
      </details>`
    : '';

  const scoreHtml = showScore
    ? `<span class="card-score">score: ${r.score.toFixed(3)}</span>`
    : '';

  const statusBadge = `<span class="status-badge" style="background:${sc.bg};color:${sc.fg};border:1px solid ${sc.border}">${statusKey}</span>`;

  const agentIdSafe = (r.id || '').replace(/'/g, "\\'");
  const historyId = `history-${(r.id || '').replace(/[^a-z0-9]/gi, '_')}`;

  return `
    <div class="agent-card" onclick="trackClick('${agentIdSafe}')">
      <div class="card-header">
        ${showScore ? `<span class="card-rank">${r._rank}</span>` : ''}
        <div>
          <div class="card-title">${c.name}</div>
          <div class="card-id">${c.humanReadableId || r.id || ''}</div>
        </div>
        <div class="card-header-right">
          ${scoreHtml}
          ${statusBadge}
        </div>
      </div>
      <div class="card-desc">${c.description}</div>
      <div class="card-meta">
        Endpoint: <a href="${c.url}" target="_blank" onclick="event.stopPropagation()">${c.url}</a>
        · v${c.agentVersion || '—'}
        ${c.provider ? `· ${c.provider.name}` : ''}
      </div>
      ${tagsHtml}
      ${skillsHtml}
      <div class="card-actions">
        <button class="btn-history" onclick="event.stopPropagation();toggleHistory('${agentIdSafe}', '${historyId}')">History</button>
      </div>
      <div id="${historyId}" class="agent-history hidden"></div>
    </div>
  `;
}


/* ---- Search ---- */
async function doSearch() {
  const query = $('query').value.trim();
  if (!query) return;

  const tagRaw = $('tag-input').value.trim();
  const tags = tagRaw ? tagRaw.split(',').map(t => t.trim()).filter(Boolean) : [];

  $('search-btn').textContent = 'Searching…';
  $('search-btn').disabled = true;

  try {
    const res = await fetch(`${API}/search`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, top_k: 8, tags }),
    });
    const data = await res.json();

    const sec = $('results-section');
    const list = $('results-list');
    sec.classList.remove('hidden');
    $('results-title').textContent =
      `${data.results.length} result${data.results.length !== 1 ? 's' : ''} for "${query}"`;

    if (!data.results.length) {
      list.innerHTML = '<p style="color:var(--muted)">No agents found. Try different keywords.</p>';
      return;
    }
    list.innerHTML = data.results
      .map((r, i) => buildAgentCard({ ...r, _rank: i + 1 }, true))
      .join('');
  } catch (e) {
    alert('Search failed: ' + e.message);
  } finally {
    $('search-btn').textContent = 'Search';
    $('search-btn').disabled = false;
  }
}

$('query').addEventListener('keydown', e => {
  if (e.key === 'Enter') doSearch();
});


/* ---- Click tracking ---- */
async function trackClick(agentId) {
  if (!agentId) return;
  try {
    await fetch(`${API}/agents/${encodeURIComponent(agentId)}/click`, { method: 'POST' });
  } catch (_) { /* best-effort */ }
}


/* ---- Version history ---- */
async function toggleHistory(agentId, containerId) {
  const el = document.getElementById(containerId);
  if (!el) return;

  if (!el.classList.contains('hidden')) {
    el.classList.add('hidden');
    el.innerHTML = '';
    return;
  }

  el.innerHTML = '<em style="color:var(--muted)">Loading history…</em>';
  el.classList.remove('hidden');

  try {
    const res = await fetch(`${API}/agents/${encodeURIComponent(agentId)}/history`);
    const data = await res.json();
    const versions = data.versions || [];

    if (!versions.length) {
      el.innerHTML = '<em style="color:var(--muted)">No version history yet.</em>';
      return;
    }

    el.innerHTML = versions.reverse().map(v => {
      const diffKeys = Object.keys(v.diff || {});
      const diffHtml = diffKeys.length
        ? diffKeys.map(field => {
            const d = v.diff[field];
            return `<div class="version-diff-field">
              <strong>${field}</strong><br>
              <span class="diff-old">− ${JSON.stringify(d.old)}</span><br>
              <span class="diff-new">+ ${JSON.stringify(d.new)}</span>
            </div>`;
          }).join('')
        : '<em style="color:var(--muted)">Initial registration</em>';

      return `<div class="version-entry">
        <div class="version-meta">v${v.version_num} · ${v.created_at}</div>
        ${diffHtml}
      </div>`;
    }).join('');
  } catch (e) {
    el.innerHTML = `<em style="color:#ff8080">Failed to load history: ${e.message}</em>`;
  }
}


/* ---- Register ---- */
function switchTab(tab) {
  $('tab-json').classList.toggle('hidden', tab !== 'json');
  $('tab-url').classList.toggle('hidden', tab !== 'url');
  document.querySelectorAll('.tab').forEach(b => b.classList.remove('active'));
  event.target.classList.add('active');
}

async function registerAgent() {
  const msgEl = $('register-msg');
  msgEl.className = '';
  msgEl.textContent = '';

  const jsonTab = !$('tab-json').classList.contains('hidden');
  let body = {};

  if (jsonTab) {
    const raw = $('card-json').value.trim();
    if (!raw) { msgEl.className = 'error'; msgEl.textContent = 'Paste an agent card JSON.'; return; }
    try {
      body = { agent_card: JSON.parse(raw) };
    } catch {
      msgEl.className = 'error'; msgEl.textContent = 'Invalid JSON.'; return;
    }
  } else {
    const url = $('agent-url').value.trim();
    if (!url) { msgEl.className = 'error'; msgEl.textContent = 'Enter an agent URL.'; return; }
    body = { agent_card: _minimalCard(), card_url: url };
  }

  const apiKey = $('api-key').value.trim();
  const headers = { 'Content-Type': 'application/json' };
  if (apiKey) headers['X-Api-Key'] = apiKey;

  try {
    const res = await fetch(`${API}/register`, {
      method: 'POST',
      headers,
      body: JSON.stringify(body),
    });
    const data = await res.json();
    if (!res.ok) {
      const detail = Array.isArray(data.detail)
        ? data.detail.map(e => `${e.loc?.slice(-1)[0] ?? 'field'}: ${e.msg}`).join('\n')
        : (data.detail || JSON.stringify(data));
      throw new Error(detail);
    }
    msgEl.className = 'success';
    msgEl.textContent = `✓ ${data.message} (id: ${data.id})`;
    loadAgents();
  } catch (e) {
    msgEl.className = 'error';
    msgEl.textContent = 'Error: ' + e.message;
  }
}

function _minimalCard() {
  return {
    schemaVersion: "1.0",
    humanReadableId: "fetched/from-url",
    name: "Fetched Agent",
    description: "",
    url: $('agent-url').value.trim(),
    agentVersion: "1.0.0",
    provider: { name: "Unknown" },
    capabilities: { a2aVersion: "1.0" },
    authSchemes: [{ type: "none" }],
  };
}


/* ---- All agents ---- */
async function loadAgents() {
  const list = $('agents-list');
  list.innerHTML = '<em>Loading…</em>';
  try {
    const res = await fetch(`${API}/agents`);
    const data = await res.json();
    const agents = data.agents ?? data;
    if (!agents.length) {
      list.innerHTML = '<em style="color:var(--muted)">No agents registered yet.</em>';
      return;
    }
    list.innerHTML = agents
      .map(r => buildAgentCard({ id: r.id, score: 1, agent_card: r.agent_card, health: r.health, _rank: '' }, false))
      .join('');
  } catch (e) {
    list.innerHTML = `<em style="color:#ff8080">Failed to load agents: ${e.message}</em>`;
  }
}


/* ---- Analytics ---- */
async function loadAnalytics() {
  const out = $('analytics-output');
  out.innerHTML = '<em style="color:var(--muted)">Loading…</em>';
  const apiKey = $('analytics-api-key').value.trim();
  const headers = {};
  if (apiKey) headers['X-Api-Key'] = apiKey;

  try {
    const res = await fetch(`${API}/analytics`, { headers });
    if (res.status === 401) {
      out.innerHTML = '<em style="color:#ff8080">Invalid API key.</em>';
      return;
    }
    const data = await res.json();
    out.innerHTML = renderAnalytics(data);
  } catch (e) {
    out.innerHTML = `<em style="color:#ff8080">Failed: ${e.message}</em>`;
  }
}

function renderAnalytics(data) {
  const topQ = (data.top_queries || [])
    .map((q, i) => `<li><code>${q.query}</code><span class="analytic-count">${q.count}</span></li>`).join('');
  const zeroQ = (data.zero_result_queries || [])
    .map(q => `<li><code>${q.query}</code><span class="analytic-count">${q.count}</span></li>`).join('');
  const topC = (data.top_clicked_agents || [])
    .map(c => `<li><code>${c.agent_id}</code><span class="analytic-count">${c.clicks} clicks</span></li>`).join('');

  return `<div class="analytics-grid">
    <div>
      <h3>Top Queries</h3>
      <ol>${topQ || '<li style="color:var(--muted)">No data yet</li>'}</ol>
    </div>
    <div>
      <h3>Zero-Result Queries</h3>
      <ol>${zeroQ || '<li style="color:var(--muted)">None — great!</li>'}</ol>
    </div>
    <div>
      <h3>Top Clicked Agents</h3>
      <ol>${topC || '<li style="color:var(--muted)">No clicks yet</li>'}</ol>
    </div>
  </div>`;
}


/* ---- Init ---- */
loadAgents();
