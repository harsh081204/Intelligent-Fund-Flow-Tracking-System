"""
Graph Visualization Layer — FastAPI + PyVis
Serves an interactive fund-flow graph in the browser.
"""

import os
import json
import random
import tempfile
from pathlib import Path
from typing import Optional

import networkx as nx
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pyvis.network import Network

from data_ingestion import ingest_data_from_parquet
from graph_engine import get_graph, validate_graph

# ─────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────
app = FastAPI(
    title="Intelligent Fund Flow — Graph Visualizer",
    description="Interactive graph visualization layer over the fund-flow transaction engine.",
    version="1.0.0",
)

# Serve static assets (dashboard HTML, CSS, JS)
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ─────────────────────────────────────────────
# Lazy-load the graph once at startup
# ─────────────────────────────────────────────
_graph: Optional[nx.MultiDiGraph] = None

def get_loaded_graph() -> nx.MultiDiGraph:
    global _graph
    if _graph is None:
        parquet_path = Path(__file__).parent / "sampled_transactions.parquet"
        if not parquet_path.exists():
            raise HTTPException(status_code=503, detail="sampled_transactions.parquet not found.")
        df = ingest_data_from_parquet(str(parquet_path))
        _graph = get_graph(df)
    return _graph


# ─────────────────────────────────────────────
# Colour helpers
# ─────────────────────────────────────────────
TX_TYPE_COLORS = {
    "TRANSFER":  "#6C63FF",
    "CASH_OUT":  "#FF6584",
    "PAYMENT":   "#43E8D8",
    "CASH_IN":   "#F9C74F",
    "DEBIT":     "#90BE6D",
}

def node_color(attrs: dict) -> str:
    if attrs.get("is_fraud"):
        return "#FF3B3B"          # bright red  — fraud
    ratio = attrs.get("total_sent", 0) / max(attrs.get("total_received", 1), 1)
    if ratio > 5:
        return "#FFA630"          # orange      — heavy sender
    return "#4ADE80"              # green        — clean / balanced

def edge_color(tx_type: str) -> str:
    return TX_TYPE_COLORS.get(tx_type, "#94A3B8")


# ─────────────────────────────────────────────
# PyVis builder
# ─────────────────────────────────────────────
def build_pyvis(
    G: nx.MultiDiGraph,
    sample_nodes: int = 200,
    fraud_only: bool = False,
    tx_type_filter: Optional[str] = None,
    step_min: int = 0,
    step_max: int = 999999,
) -> str:
    """
    Creates a PyVis HTML string for the (sampled) subgraph.
    Returns the raw HTML as a string.
    """
    net = Network(
        height="100%",
        width="100%",
        directed=True,
        bgcolor="#0F172A",
        font_color="#E2E8F0",
        notebook=False,
    )
    net.set_options("""
    {
      "nodes": {
        "borderWidth": 2,
        "shadow": { "enabled": true, "size": 10 },
        "font": { "size": 11, "face": "Inter, sans-serif" }
      },
      "edges": {
        "smooth": { "type": "curvedCW", "roundness": 0.2 },
        "arrows": { "to": { "enabled": true, "scaleFactor": 0.6 } },
        "shadow": { "enabled": true }
      },
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -50,
          "centralGravity": 0.01,
          "springLength": 120,
          "springConstant": 0.08,
          "damping": 0.9
        },
        "solver": "forceAtlas2Based",
        "stabilization": { "iterations": 150 }
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 100,
        "navigationButtons": true,
        "keyboard": true
      }
    }
    """)

    # ── Filter candidate nodes ──────────────────
    if fraud_only:
        candidate_nodes = [n for n, d in G.nodes(data=True) if d.get("is_fraud")]
    else:
        candidate_nodes = list(G.nodes())

    if len(candidate_nodes) > sample_nodes:
        candidate_nodes = random.sample(candidate_nodes, sample_nodes)

    node_set = set(candidate_nodes)

    # ── Add nodes ──────────────────────────────
    for node in candidate_nodes:
        attrs = G.nodes[node]
        color = node_color(attrs)
        size  = min(10 + (attrs.get("tx_count", 1) ** 0.5) * 2, 50)
        title = (
            f"<div style='font-family:Inter,sans-serif;min-width:180px'>"
            f"<b style='color:{color}'>{node}</b><br>"
            f"<hr style='border-color:#334155;margin:4px 0'>"
            f"💸 Sent:     <b>${attrs.get('total_sent', 0):,.0f}</b><br>"
            f"💰 Received: <b>${attrs.get('total_received', 0):,.0f}</b><br>"
            f"🔢 Tx Count: <b>{attrs.get('tx_count', 0)}</b><br>"
            f"⏱ Steps:    <b>{attrs.get('first_seen', '?')} → {attrs.get('last_seen', '?')}</b><br>"
            f"🏷 Types:    <b>{', '.join(attrs.get('tx_types', []))}</b><br>"
            f"🚨 Fraud:    <b>{'YES' if attrs.get('is_fraud') else 'No'}</b>"
            f"</div>"
        )
        net.add_node(
            node,
            label=node[:8] + "…" if len(node) > 8 else node,
            title=title,
            color=color,
            size=size,
            borderColor="#FFFFFF" if attrs.get("is_fraud") else color,
        )

    # ── Add edges ──────────────────────────────
    for u, v, data in G.edges(data=True):
        if u not in node_set or v not in node_set:
            continue
        step = data.get("step", 0)
        if not (step_min <= step <= step_max):
            continue
        tx_type = data.get("tx_type", "")
        if tx_type_filter and tx_type != tx_type_filter:
            continue

        amount = data.get("amount", 0)
        width  = min(1 + (amount ** 0.3) / 10, 8)
        color  = "#FF3B3B" if data.get("is_fraud") else edge_color(tx_type)
        title  = (
            f"<div style='font-family:Inter,sans-serif'>"
            f"<b>{tx_type}</b> @ step {step}<br>"
            f"Amount: <b>${amount:,.2f}</b><br>"
            f"Fraud:  <b>{'YES 🚨' if data.get('is_fraud') else 'No'}</b>"
            f"</div>"
        )
        net.add_edge(u, v, title=title, color=color, width=width)

    # Return raw HTML string
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as f:
        tmp_path = f.name
    net.save_graph(tmp_path)
    html = Path(tmp_path).read_text()
    os.unlink(tmp_path)
    return html


# ─────────────────────────────────────────────
# API Routes
# ─────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, tags=["Dashboard"])
async def dashboard():
    """Serve the main visualization dashboard."""
    return HTMLResponse(content=DASHBOARD_HTML)


@app.get("/api/graph", response_class=HTMLResponse, tags=["Graph"])
async def graph_view(
    sample: int   = Query(200,  ge=10, le=2000, description="Max nodes to render"),
    fraud_only: bool = Query(False, description="Show only fraud-involved nodes"),
    tx_type: Optional[str] = Query(None, description="Filter by tx type: TRANSFER | CASH_OUT"),
    step_min: int = Query(0,   ge=0,  description="Min transaction step"),
    step_max: int = Query(999999, description="Max transaction step"),
):
    """Returns a self-contained PyVis HTML graph."""
    G = get_loaded_graph()
    html = build_pyvis(G, sample_nodes=sample, fraud_only=fraud_only,
                       tx_type_filter=tx_type, step_min=step_min, step_max=step_max)
    return HTMLResponse(content=html)


@app.get("/api/stats", tags=["Graph"])
async def graph_stats():
    """Returns high-level graph statistics."""
    G = get_loaded_graph()
    fraud_nodes = [n for n, d in G.nodes(data=True) if d.get("is_fraud")]
    fraud_edges = [(u, v, d) for u, v, d in G.edges(data=True) if d.get("is_fraud")]

    return JSONResponse({
        "total_nodes": G.number_of_nodes(),
        "total_edges": G.number_of_edges(),
        "fraud_nodes": len(fraud_nodes),
        "fraud_edges": len(fraud_edges),
        "weakly_connected_components": nx.number_weakly_connected_components(G),
        "avg_degree": round(
            sum(d for _, d in G.degree()) / max(G.number_of_nodes(), 1), 2
        ),
        "top_senders": _top_nodes(G, "total_sent", 5),
        "top_receivers": _top_nodes(G, "total_received", 5),
    })


@app.get("/api/node/{node_id}", tags=["Graph"])
async def node_detail(node_id: str):
    """Returns full attributes for a single node."""
    G = get_loaded_graph()
    if node_id not in G:
        raise HTTPException(status_code=404, detail=f"Node '{node_id}' not found.")
    attrs = dict(G.nodes[node_id])
    neighbors_out = list(G.successors(node_id))
    neighbors_in  = list(G.predecessors(node_id))
    return JSONResponse({
        "node_id": node_id,
        "attributes": attrs,
        "out_neighbors": neighbors_out[:20],
        "in_neighbors":  neighbors_in[:20],
        "out_degree": G.out_degree(node_id),
        "in_degree":  G.in_degree(node_id),
    })


@app.get("/api/health", tags=["System"])
async def health():
    return {"status": "ok", "graph_loaded": _graph is not None}


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def _top_nodes(G: nx.MultiDiGraph, attr: str, n: int) -> list:
    ranked = sorted(
        [(node, G.nodes[node].get(attr, 0)) for node in G.nodes()],
        key=lambda x: x[1], reverse=True
    )
    return [{"node": r[0], attr: round(r[1], 2)} for r in ranked[:n]]


# ─────────────────────────────────────────────
# Embedded Dashboard HTML
# ─────────────────────────────────────────────
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Fund Flow Graph Visualizer</title>
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet" />
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    :root {
      --bg:       #0F172A;
      --surface:  #1E293B;
      --border:   #334155;
      --text:     #E2E8F0;
      --muted:    #94A3B8;
      --accent:   #6C63FF;
      --fraud:    #FF3B3B;
      --green:    #4ADE80;
      --orange:   #FFA630;
    }

    body { font-family: 'Inter', sans-serif; background: var(--bg); color: var(--text);
           display: flex; flex-direction: column; height: 100vh; overflow: hidden; }

    /* ── Header ── */
    header {
      display: flex; align-items: center; justify-content: space-between;
      padding: 0 24px; height: 56px; background: var(--surface);
      border-bottom: 1px solid var(--border); flex-shrink: 0;
      box-shadow: 0 2px 16px rgba(0,0,0,0.4);
    }
    header h1 { font-size: 16px; font-weight: 600; letter-spacing: -0.3px;
                display: flex; align-items: center; gap: 10px; }
    header h1 span.logo { font-size: 22px; }
    .badge { background: var(--accent); color: #fff; font-size: 10px;
             font-weight: 600; padding: 2px 8px; border-radius: 50px; letter-spacing: 0.5px; }
    #status-dot { width: 8px; height: 8px; border-radius: 50%; background: #64748B;
                  transition: background 0.3s; }
    #status-dot.ready { background: var(--green); box-shadow: 0 0 8px var(--green); }
    #status-dot.loading { background: var(--orange); box-shadow: 0 0 8px var(--orange); }

    /* ── Layout ── */
    .layout { display: flex; flex: 1; overflow: hidden; }

    /* ── Sidebar ── */
    aside {
      width: 280px; background: var(--surface); border-right: 1px solid var(--border);
      display: flex; flex-direction: column; flex-shrink: 0; overflow-y: auto;
    }
    .sidebar-section { padding: 16px; border-bottom: 1px solid var(--border); }
    .sidebar-section h2 { font-size: 11px; font-weight: 600; color: var(--muted);
                          text-transform: uppercase; letter-spacing: 1px; margin-bottom: 12px; }
    label { display: block; font-size: 12px; color: var(--muted); margin-bottom: 4px; margin-top: 10px; }
    label:first-of-type { margin-top: 0; }

    input[type=range] { width: 100%; accent-color: var(--accent); cursor: pointer; }
    input[type=number], select {
      width: 100%; background: var(--bg); border: 1px solid var(--border);
      color: var(--text); border-radius: 6px; padding: 6px 10px; font-size: 13px;
      font-family: 'Inter', sans-serif; outline: none;
      transition: border-color 0.2s;
    }
    input[type=number]:focus, select:focus { border-color: var(--accent); }
    .range-val { font-size: 13px; font-weight: 600; color: var(--text); float: right; }

    .toggle-row { display: flex; align-items: center; justify-content: space-between;
                  margin-top: 10px; }
    .toggle-row span { font-size: 13px; }
    .toggle { position: relative; width: 40px; height: 22px; }
    .toggle input { opacity: 0; width: 0; height: 0; }
    .slider-tog { position: absolute; inset: 0; background: var(--border);
                  border-radius: 22px; cursor: pointer; transition: background 0.2s; }
    .slider-tog::before { content: ''; position: absolute; height: 16px; width: 16px;
                           left: 3px; top: 3px; background: white; border-radius: 50%;
                           transition: transform 0.2s; }
    .toggle input:checked + .slider-tog { background: var(--accent); }
    .toggle input:checked + .slider-tog::before { transform: translateX(18px); }

    .btn-render {
      width: 100%; padding: 10px; background: var(--accent); color: #fff;
      border: none; border-radius: 8px; font-size: 13px; font-weight: 600;
      cursor: pointer; transition: opacity 0.2s, transform 0.1s;
      font-family: 'Inter', sans-serif; letter-spacing: 0.2px;
    }
    .btn-render:hover { opacity: 0.88; }
    .btn-render:active { transform: scale(0.97); }

    /* ── Stats Cards ── */
    .stats-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; padding: 16px; }
    .stat-card {
      background: var(--bg); border: 1px solid var(--border); border-radius: 8px;
      padding: 10px 12px;
    }
    .stat-card .val { font-size: 18px; font-weight: 700; }
    .stat-card .lbl { font-size: 10px; color: var(--muted); text-transform: uppercase;
                      letter-spacing: 0.5px; margin-top: 2px; }
    .stat-card.fraud .val { color: var(--fraud); }
    .stat-card.clean .val { color: var(--green); }
    .stat-card.accent .val { color: var(--accent); }
    .stat-card.orange .val { color: var(--orange); }

    /* ── Legend ── */
    .legend { padding: 16px; }
    .legend h2 { font-size: 11px; font-weight: 600; color: var(--muted);
                 text-transform: uppercase; letter-spacing: 1px; margin-bottom: 10px; }
    .legend-item { display: flex; align-items: center; gap: 8px;
                   font-size: 12px; margin-bottom: 6px; }
    .dot { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }
    .line { width: 20px; height: 3px; border-radius: 2px; flex-shrink: 0; }

    /* ── Main Graph Area ── */
    main { flex: 1; position: relative; overflow: hidden; }
    #graph-frame {
      width: 100%; height: 100%; border: none; background: var(--bg);
    }
    #overlay {
      position: absolute; inset: 0; display: flex; flex-direction: column;
      align-items: center; justify-content: center; gap: 16px;
      background: rgba(15, 23, 42, 0.92); backdrop-filter: blur(4px);
      pointer-events: none; transition: opacity 0.4s;
    }
    #overlay.hidden { opacity: 0; }
    .spinner {
      width: 44px; height: 44px; border: 3px solid var(--border);
      border-top-color: var(--accent); border-radius: 50%;
      animation: spin 0.8s linear infinite;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
    #overlay p { color: var(--muted); font-size: 14px; }
    #overlay .hint { font-size: 12px; color: var(--border); }
  </style>
</head>
<body>

<!-- ── Header ── -->
<header>
  <h1>
    <span class="logo">🕸️</span>
    Fund Flow Graph Visualizer
    <span class="badge">LIVE</span>
  </h1>
  <div style="display:flex;align-items:center;gap:10px">
    <span id="status-text" style="font-size:12px;color:var(--muted)">Loading graph…</span>
    <div id="status-dot"></div>
  </div>
</header>

<div class="layout">

  <!-- ── Sidebar ── -->
  <aside>

    <!-- Controls -->
    <div class="sidebar-section">
      <h2>Graph Controls</h2>

      <label>Sample Nodes <span class="range-val" id="sample-val">200</span></label>
      <input type="range" id="sample" min="20" max="1000" step="20" value="200"
             oninput="document.getElementById('sample-val').textContent = this.value" />

      <label>Transaction Type</label>
      <select id="tx-type">
        <option value="">All Types</option>
        <option value="TRANSFER">TRANSFER</option>
        <option value="CASH_OUT">CASH_OUT</option>
      </select>

      <label>Step Min</label>
      <input type="number" id="step-min" value="0" min="0" />

      <label>Step Max</label>
      <input type="number" id="step-max" value="999999" min="0" />

      <div class="toggle-row" style="margin-top:14px">
        <span>Fraud Nodes Only</span>
        <label class="toggle">
          <input type="checkbox" id="fraud-only" />
          <span class="slider-tog"></span>
        </label>
      </div>

      <button class="btn-render" style="margin-top:16px" onclick="renderGraph()">
        ⚡ Render Graph
      </button>
    </div>

    <!-- Stats -->
    <div class="sidebar-section" style="padding-bottom:0;border-bottom:none">
      <h2>Graph Statistics</h2>
    </div>
    <div class="stats-grid" id="stats-grid">
      <div class="stat-card accent"><div class="val" id="s-nodes">—</div><div class="lbl">Nodes</div></div>
      <div class="stat-card accent"><div class="val" id="s-edges">—</div><div class="lbl">Edges</div></div>
      <div class="stat-card fraud"><div class="val" id="s-fnodes">—</div><div class="lbl">Fraud Nodes</div></div>
      <div class="stat-card fraud"><div class="val" id="s-fedges">—</div><div class="lbl">Fraud Edges</div></div>
      <div class="stat-card orange"><div class="val" id="s-comps">—</div><div class="lbl">Components</div></div>
      <div class="stat-card clean"><div class="val" id="s-deg">—</div><div class="lbl">Avg Degree</div></div>
    </div>

    <!-- Legend -->
    <div class="legend">
      <h2>Legend</h2>
      <div class="legend-item"><div class="dot" style="background:#FF3B3B"></div> Fraud Account</div>
      <div class="legend-item"><div class="dot" style="background:#4ADE80"></div> Clean Account</div>
      <div class="legend-item"><div class="dot" style="background:#FFA630"></div> Heavy Sender</div>
      <div class="legend-item" style="margin-top:8px">
        <div class="line" style="background:#6C63FF"></div> TRANSFER
      </div>
      <div class="legend-item">
        <div class="line" style="background:#FF6584"></div> CASH_OUT
      </div>
      <div class="legend-item">
        <div class="line" style="background:#FF3B3B"></div> Fraud Edge
      </div>
      <div style="margin-top:12px;font-size:11px;color:var(--muted);line-height:1.6">
        Node size ∝ transaction count<br>
        Edge width ∝ amount
      </div>
    </div>

  </aside>

  <!-- ── Main Graph Canvas ── -->
  <main>
    <div id="overlay">
      <div class="spinner"></div>
      <p>Building graph…</p>
      <span class="hint">This may take a moment on first load</span>
    </div>
    <iframe id="graph-frame" title="Fund Flow Graph"></iframe>
  </main>

</div>

<script>
  async function loadStats() {
    try {
      const res = await fetch('/api/stats');
      const d   = await res.json();
      document.getElementById('s-nodes').textContent  = d.total_nodes.toLocaleString();
      document.getElementById('s-edges').textContent  = d.total_edges.toLocaleString();
      document.getElementById('s-fnodes').textContent = d.fraud_nodes.toLocaleString();
      document.getElementById('s-fedges').textContent = d.fraud_edges.toLocaleString();
      document.getElementById('s-comps').textContent  = d.weakly_connected_components.toLocaleString();
      document.getElementById('s-deg').textContent    = d.avg_degree;
      document.getElementById('status-text').textContent = 'Graph ready';
      document.getElementById('status-dot').className = 'ready';
    } catch(e) {
      document.getElementById('status-text').textContent = 'Error loading stats';
    }
  }

  async function renderGraph() {
    const sample    = document.getElementById('sample').value;
    const txType    = document.getElementById('tx-type').value;
    const stepMin   = document.getElementById('step-min').value;
    const stepMax   = document.getElementById('step-max').value;
    const fraudOnly = document.getElementById('fraud-only').checked;

    const overlay = document.getElementById('overlay');
    overlay.classList.remove('hidden');
    document.getElementById('status-dot').className = 'loading';
    document.getElementById('status-text').textContent = 'Rendering…';

    const params = new URLSearchParams({
      sample, step_min: stepMin, step_max: stepMax, fraud_only: fraudOnly,
      ...(txType ? { tx_type: txType } : {})
    });

    const frame = document.getElementById('graph-frame');
    frame.src = '/api/graph?' + params.toString();
    frame.onload = () => {
      overlay.classList.add('hidden');
      document.getElementById('status-dot').className = 'ready';
      document.getElementById('status-text').textContent = `Showing ${sample} nodes`;
    };
  }

  // Init
  loadStats();
  renderGraph();
</script>
</body>
</html>
"""


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("graph_visualizer:app", host="0.0.0.0", port=8000, reload=False)
