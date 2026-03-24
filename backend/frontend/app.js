/* ============================================================
   Agent Search Engine — frontend JS
   ============================================================ */

const API = '';   // same-origin; change to 'http://localhost:8000' for local dev


/* ---- Utilities ---- */
function $(id) { return document.getElementById(id); }

function buildAgentCard(r, showScore = true) {
  const c = r.agent_card;
  const tags = [...(c.tags || [])];
  const skills = c.skills || [];

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

  return `
    <div class="agent-card">
      <div class="card-header">
        ${showScore ? `<span class="card-rank">${r._rank}</span>` : ''}
        <div>
          <div class="card-title">${c.name}</div>
          <div class="card-id">${c.humanReadableId}</div>
        </div>
        ${scoreHtml}
      </div>
      <div class="card-desc">${c.description}</div>
      <div class="card-meta">
        Endpoint: <a href="${c.url}" target="_blank">${c.url}</a>
        · v${c.agentVersion || '—'}
        ${c.provider ? `· ${c.provider.name}` : ''}
      </div>
      ${tagsHtml}
      ${skillsHtml}
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

  try {
    const res = await fetch(`${API}/register`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const data = await res.json();
    if (!res.ok) {
      // FastAPI validation errors come back as an array in data.detail
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
  // placeholder — server will overwrite from the fetched URL
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
    if (!data.length) {
      list.innerHTML = '<em style="color:var(--muted)">No agents registered yet.</em>';
      return;
    }
    list.innerHTML = data
      .map(r => buildAgentCard({ id: r.id, score: 1, agent_card: r.agent_card, _rank: '' }, false))
      .join('');
  } catch (e) {
    list.innerHTML = `<em style="color:#ff8080">Failed to load agents: ${e.message}</em>`;
  }
}


/* ---- Init ---- */
loadAgents();
