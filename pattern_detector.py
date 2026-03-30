import networkx as nx
import pandas as pd
import numpy as np
from collections import defaultdict
import time
import os
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from data_ingestion import ingest_data_from_parquet

# Import our optimized graph builder
from graph_engine import build_transaction_graph, validate_graph, get_graph, update_graph_with_new_data

# ==========================================
# LAYER 1: ALGORITHMIC PATTERN DETECTORS
# ==========================================

def detect_round_trips(G, max_cycle_length=6, min_amount=1000):
    """
    Find closed loops where money returns to origin account.
    """
    suspicious = []
    
    # ⚠️ CRITICAL: Filter to high-degree nodes first to avoid memory explosion
    high_degree_nodes = [
        n for n in G.nodes() 
        if G.out_degree(n) >= 2 and G.in_degree(n) >= 1
    ]
    subG = G.subgraph(high_degree_nodes)
    
    try:
        for cycle in nx.simple_cycles(subG):
            if 2 <= len(cycle) <= max_cycle_length:
                total_amount = 0
                valid = True
                steps = []
                
                for i in range(len(cycle)):
                    src = cycle[i]
                    dst = cycle[(i + 1) % len(cycle)]
                    
                    if G.has_edge(src, dst):
                        # Sum all edges between pair (MultiDiGraph)
                        edges = G[src][dst]
                        total_amount += sum(e['amount'] for e in edges.values())
                        steps.extend([e['step'] for e in edges.values()])
                    else:
                        valid = False
                        break
                        
                if valid and total_amount >= min_amount:
                    time_span = max(steps) - min(steps) if steps else 0
                    
                    suspicious.append({
                        'pattern': 'ROUND_TRIP',
                        'accounts': cycle,
                        'cycle_length': len(cycle),
                        'total_amount': total_amount,
                        'time_span_hrs': time_span,
                        'risk_score': min(100, 40 + (len(cycle) * 10) + (20 if time_span < 24 else 0))
                    })
    except Exception as e:
        print(f"Cycle detection error: {e}")
        
    return suspicious

def detect_layering(G, min_fanout=5, time_window=48, min_amount=50000):
    """
    Detect: 1 source → many intermediaries → 1 destination (Money Mule structure)
    """
    suspicious = []
    
    for node in G.nodes():
        successors = list(G.successors(node))
        if len(successors) < min_fanout:
            continue
            
        total_out = sum(sum(e['amount'] for e in G[node][s].values()) for s in successors)
        if total_out < min_amount:
            continue
            
        out_steps = []
        for s in successors:
            for e in G[node][s].values():
                out_steps.append(e['step'])
                
        if not out_steps or (max(out_steps) - min(out_steps) > time_window):
            continue
            
        successor_destinations = defaultdict(float)
        for intermediary in successors:
            for dest in G.successors(intermediary):
                if dest != node:
                    for e in G[intermediary][dest].values():
                        successor_destinations[dest] += e['amount']
                        
        if not successor_destinations:
            continue
            
        top_dest = max(successor_destinations, key=successor_destinations.get)
        top_amount = successor_destinations[top_dest]
        convergence_ratio = top_amount / total_out if total_out > 0 else 0
        
        if convergence_ratio >= 0.5:
            suspicious.append({
                'pattern': 'LAYERING',
                'source': node,
                'intermediaries': successors,
                'final_dest': top_dest,
                'num_layers': len(successors),
                'total_amount': total_out,
                'convergence_ratio': round(convergence_ratio, 2),
                'time_span_hrs': max(out_steps) - min(out_steps),
                'risk_score': min(100, 50 + (len(successors) * 5) + int(convergence_ratio * 20))
            })
            
    return suspicious

def detect_structuring(df, threshold=200000, window_hrs=24, min_txns=3, band_low=0.75, band_high=0.99):
    """
    Detect transactions just below reporting threshold.
    """
    suspicious = []
    lower = threshold * band_low
    upper = threshold * band_high
    
    band_df = df[(df['amount'] >= lower) & (df['amount'] < upper)].copy()
    if band_df.empty:
        return suspicious
        
    band_df = band_df.sort_values(['nameOrig', 'step'])
    
    for account, group in band_df.groupby('nameOrig'):
        steps = group['step'].values
        amounts = group['amount'].values
        
        i = 0
        while i < len(steps):
            j = i
            while j < len(steps) and (steps[j] - steps[i]) <= window_hrs:
                j += 1
                
            cluster_size = j - i
            if cluster_size >= min_txns:
                cluster_amounts = amounts[i:j]
                cluster_steps = steps[i:j]
                time_span = int(cluster_steps[-1] - cluster_steps[0])
                
                suspicious.append({
                    'pattern': 'STRUCTURING',
                    'account': account,
                    'txn_count': cluster_size,
                    'total_amount': float(sum(cluster_amounts)),
                    'amounts': cluster_amounts.tolist(),
                    'time_span': time_span,
                    'threshold': threshold,
                    'risk_score': min(100, 40 + (cluster_size * 8) + (15 if time_span < 12 else 0))
                })
            i += 1
            
    return suspicious

def detect_dormant_activation(df, G, dormant_threshold=30, high_value=50000, rapid_outflow_window=48):
    """
    Detect accounts inactive for 30+ steps that suddenly receive a large transfer.
    """
    suspicious = []
    account_timeline = df.groupby('nameDest').agg(steps=('step', list), amounts=('amount', list)).to_dict('index')
    
    for account, data in account_timeline.items():
        steps = sorted(data['steps'])
        amounts = data['amounts']
        
        if len(steps) < 2:
            continue
            
        gaps = [(steps[i+1] - steps[i], i) for i in range(len(steps)-1)]
        max_gap, gap_idx = max(gaps, key=lambda x: x[0])
        
        if max_gap < dormant_threshold:
            continue
            
        reactivation_step = steps[gap_idx + 1]
        reactivation_amount = amounts[gap_idx + 1]
        
        if reactivation_amount < high_value:
            continue
            
        rapid_outflow = 0
        if account in G:
            for dest in G.successors(account):
                for edge in G[account][dest].values():
                    if reactivation_step <= edge['step'] <= reactivation_step + rapid_outflow_window:
                        rapid_outflow += edge['amount']
                        
        outflow_ratio = rapid_outflow / reactivation_amount if reactivation_amount > 0 else 0
        
        suspicious.append({
            'pattern': 'DORMANT_ACTIVATION',
            'account': account,
            'dormant_gap_hrs': int(max_gap),
            'reactivation_amount': float(reactivation_amount),
            'rapid_outflow': float(rapid_outflow),
            'outflow_ratio': round(outflow_ratio, 2),
            'risk_score': min(100, 50 + (20 if max_gap > 100 else 10) + (30 if outflow_ratio > 0.8 else 15 if outflow_ratio > 0.5 else 0))
        })
        
    return suspicious

# ==========================================
# LAYER 2: ISOLATION FOREST ML SCORER
# ==========================================

def extract_node_features(G):
    """
    Extract ML features for every node from the graph structure.
    """
    features = {}
    
    # Betweenness on high degree nodes only
    high_degree = sorted(G.nodes(), key=lambda n: G.degree(n), reverse=True)[:5000]
    subG = G.subgraph(high_degree)
    betweenness = nx.betweenness_centrality(subG, normalized=True)
    
    for node in G.nodes():
        in_deg = G.in_degree(node)
        out_deg = G.out_degree(node)
        
        in_amounts = [e['amount'] for u in G.predecessors(node) for e in G[u][node].values()]
        out_amounts = [e['amount'] for v in G.successors(node) for e in G[node][v].values()]
        
        total_in = sum(in_amounts) if in_amounts else 0
        total_out = sum(out_amounts) if out_amounts else 0
        
        all_steps = [e['step'] for u in G.predecessors(node) for e in G[u][node].values()] + \
                    [e['step'] for v in G.successors(node) for e in G[node][v].values()]
                    
        features[node] = {
            'in_degree': in_deg,
            'out_degree': out_deg,
            'degree_ratio': out_deg / (in_deg + 1),
            'total_degree': in_deg + out_deg,
            'total_received': total_in,
            'total_sent': total_out,
            'amount_flow_ratio': total_out / (total_in + 1),
            'max_single_txn': max(in_amounts + out_amounts) if (in_amounts or out_amounts) else 0,
            'avg_txn_amount': np.mean(in_amounts + out_amounts) if (in_amounts or out_amounts) else 0,
            'txn_velocity': (in_deg + out_deg) / (max(all_steps) - min(all_steps) + 1) if len(all_steps) > 1 else 0,
            'betweenness': betweenness.get(node, 0),
            'unique_counterparties': len(set(list(G.predecessors(node)) + list(G.successors(node)))),
            'balance_before': G.nodes[node].get('oldbalanceOrg', 0),
            'balance_after': G.nodes[node].get('newbalanceOrig', 0),
            'balance_drain': max(0, G.nodes[node].get('oldbalanceOrg', 0) - G.nodes[node].get('newbalanceOrig', 0))
        }
        
    return pd.DataFrame(features).T.fillna(0)

def get_model(features_df, force_retrain=False, contamination=0.05):
    """
    Train or load Isolation Forest model.
    """
    MODEL_PATH = "models/isolation_forest.pkl"
    SCALER_PATH = "models/scaler.pkl"
    
    if os.path.exists(MODEL_PATH) and not force_retrain:
        print("Loading saved model...")
        iso = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        
        X = scaler.transform(features_df.reindex(columns=scaler.feature_names_in_, fill_value=0))
        raw_scores = iso.decision_function(X)
        min_s, max_s = raw_scores.min(), raw_scores.max()
        
        if max_s == min_s:
            ml_scores = np.zeros_like(raw_scores)
        else:
            ml_scores = (1 - (raw_scores - min_s) / (max_s - min_s + 1e-9)) * 100
            
        return dict(zip(features_df.index, ml_scores)), scaler, iso

    print("Training model...")
    scaler = StandardScaler()
    X = scaler.fit_transform(features_df)
    
    iso = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        max_samples='auto',
        random_state=42,
        n_jobs=-1
    )
    iso.fit(X)
    
    os.makedirs("models", exist_ok=True)
    joblib.dump(iso, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print("Model saved.")
    
    raw_scores = iso.decision_function(X)
    min_s, max_s = raw_scores.min(), raw_scores.max()
    
    if max_s == min_s: 
        ml_scores = np.zeros_like(raw_scores)
    else:
        # Normalize to 0-100 (invert so more negative/anomalous == higher score)
        ml_scores = (1 - (raw_scores - min_s) / (max_s - min_s + 1e-9)) * 100
        
    return dict(zip(features_df.index, ml_scores)), scaler, iso

# ==========================================
# ENSEMBLE SCORER
# ==========================================

def build_explanation(node, ml_score, patterns, flags):
    parts = []
    if ml_score > 60:
        parts.append(f"ML anomalous (score {ml_score:.0f}/100)")
    for p in patterns:
        parts.append({
            'ROUND_TRIP': "circular fund movement",
            'LAYERING': "layering hub",
            'STRUCTURING': "structured transactions below threshold",
            'DORMANT_ACTIVATION': "dormant account reactivated with high-value transfer",
        }.get(p, p))
    parts.extend(flags)
    return "; ".join(parts) if parts else "Low-level anomaly"

def compute_ensemble_risk_scores(G, ml_scores, all_patterns):
    """
    Combine ML anomaly scores with rule-based pattern detections.
    Returns a dictionary mapping node IDs to their comprehensive risk profile.
    """
    node_profiles = {}
    
    # Map patterns to the nodes involved
    node_to_patterns = defaultdict(list)
    for p in all_patterns:
        accounts = p.get('accounts') or p.get('intermediaries') or [p.get('account')] or [p.get('source')]
        for acc in filter(None, accounts):
            node_to_patterns[acc].append(p)
            
    for node in G.nodes():
        ml_score = ml_scores.get(node, 0)
        final_score = ml_score * 0.5
        
        pattern_boost = 0
        triggered_patterns = []
        for p in node_to_patterns.get(node, []):
            ptype = p['pattern']
            boost = {'ROUND_TRIP': 30, 'LAYERING': 35, 'STRUCTURING': 25, 'DORMANT_ACTIVATION': 25}.get(ptype, 10)
            pattern_boost += boost
            if ptype not in triggered_patterns:
                triggered_patterns.append(ptype)
                
        flag_boost = 0
        flags = []
        out_deg = G.out_degree(node)
        in_deg = G.in_degree(node)
        
        if out_deg >= 10:
            flag_boost += 15
            flags.append(f"high fan-out ({out_deg})")
        if out_deg > 0 and in_deg == 0:
            flag_boost += 10
            flags.append("sends only")
            
        for neighbor in G.successors(node):
            for edge in G[node][neighbor].values():
                if edge.get('is_fraud'):
                    flag_boost += 20
                    flags.append("connected to confirmed fraud")
                    break
                    
        final_score = min(100, final_score + pattern_boost + flag_boost)
        
        if final_score >= 80: severity = "CRITICAL"
        elif final_score >= 60: severity = "HIGH"
        elif final_score >= 40: severity = "MEDIUM"
        else: severity = "LOW"
        
        node_profiles[node] = {
            'node': node,
            'ml_base_score': round(ml_score, 1),
            'pattern_boost': pattern_boost,
            'flag_boost': flag_boost,
            'final_score': round(final_score, 1),
            'severity': severity,
            'triggered_patterns': triggered_patterns,
            'flags': flags,
            'explanation': build_explanation(node, ml_score, triggered_patterns, flags)
        }
        
    return node_profiles

# ==========================================
# INCREMENTAL DETECTIONS FOR NEW DATA
# ==========================================

def handle_new_data(new_df, existing_features_df, iso_model, scaler, retrain_threshold=0.10):
    new_G = build_transaction_graph(new_df)
    new_features = extract_node_features(new_G)
    
    new_ratio = len(new_features) / len(existing_features_df) if len(existing_features_df) > 0 else 1.0
    
    if new_ratio > retrain_threshold:
        print(f"New data is {new_ratio:.0%} of original — retraining")
        combined_features = pd.concat([existing_features_df, new_features]).drop_duplicates()
        ml_scores, scaler, iso_model = get_model(combined_features, force_retrain=True)
    else:
        print(f"Small batch ({new_ratio:.0%}) — scoring without retraining")
        X_new = scaler.transform(new_features.reindex(columns=scaler.feature_names_in_, fill_value=0))
        raw_scores = iso_model.decision_function(X_new)
        min_s, max_s = raw_scores.min(), raw_scores.max()
        if max_s == min_s:
            scores_array = np.zeros_like(raw_scores)
        else:
            scores_array = (1 - (raw_scores - min_s) / (max_s - min_s + 1e-9)) * 100
        ml_scores = dict(zip(new_features.index, scores_array))
        
    return ml_scores, iso_model, scaler

def run_incremental_detection(G, new_df, existing_alerts, iso_model, scaler):
    """
    Run detection only on nodes touched by new transactions.
    """
    affected_nodes = set(new_df['nameOrig'].tolist() + new_df['nameDest'].tolist())
    
    extended_nodes = set(affected_nodes)
    for node in affected_nodes:
        if node in G:
            extended_nodes.update(G.predecessors(node))
            extended_nodes.update(G.successors(node))
            for neighbor in G.successors(node):
                extended_nodes.update(G.successors(neighbor))
                
    subG = G.subgraph(extended_nodes).copy()
    
    new_patterns = []
    new_patterns += detect_round_trips(subG)
    new_patterns += detect_layering(subG)
    new_patterns += detect_structuring(new_df[new_df['nameOrig'].isin(affected_nodes)])
    new_patterns += detect_dormant_activation(new_df, subG)
    
    new_features = extract_node_features(subG)
    X_new = scaler.transform(new_features.reindex(columns=scaler.feature_names_in_, fill_value=0))
    raw_scores = iso_model.decision_function(X_new)
    min_s, max_s = raw_scores.min(), raw_scores.max()
    
    if max_s == min_s:
        raw_scores_norm = np.zeros_like(raw_scores)
    else:
        raw_scores_norm = (1 - (raw_scores - min_s) / (max_s - min_s + 1e-9)) * 100
        
    ml_scores = dict(zip(new_features.index, raw_scores_norm))
    
    new_alerts = compute_ensemble_risk_scores(subG, ml_scores, new_patterns)
    high_critical_alerts = [alert for alert in new_alerts.values() if alert['severity'] in ('HIGH', 'CRITICAL')]
    
    # Merge carefully
    all_alerts = existing_alerts.copy() if hasattr(existing_alerts, "copy") else existing_alerts
    if isinstance(all_alerts, list):
        all_alerts.extend(high_critical_alerts)
    else: # If standard dict return
        all_alerts.update({k: v for k, v in new_alerts.items() if v['severity'] in ('HIGH', 'CRITICAL')})
        
    return all_alerts, new_patterns

# ==========================================
# MAIN EXECUTION AND EVALUATION
# ==========================================

if __name__ == "__main__":
    print("1. Loading dataset...")
    df = ingest_data_from_parquet('sampled_transactions.parquet')
    
    print("2. Building or loading graph using Graph Engine...")
    G = get_graph(df)
    
    print("3. Executing Layer 1: Algorithmic Detectors...")
    # NOTE: In a true production loop these alerts would be cached too,
    # but we compute them fresh here for full-dataset logic
    t0 = time.time()
    round_trips = detect_round_trips(G)
    layering = detect_layering(G)
    structuring = detect_structuring(df)
    dormant = detect_dormant_activation(df, G)
    print(f"Layer 1 finished in {time.time() - t0:.2f}s")
    
    all_patterns = round_trips + layering + structuring + dormant
    
    print("4. Executing Layer 2: Feature Extraction & Isolation Forest...")
    t0 = time.time()
    # Cache features if needed, here we re-extract or load from model step
    features_df = extract_node_features(G)
    ml_scores, scaler, iso_model = get_model(features_df)
    print(f"Layer 2 finished in {time.time() - t0:.2f}s")
    
    print("5. Computing Ensemble Scores...")
    ensemble_scores = compute_ensemble_risk_scores(G, ml_scores, all_patterns)
    
    # ----------------------------------------------------
    # REPORTING AND GROUND TRUTH EVALUATION
    # ----------------------------------------------------
    print("\n--- DETECTION SUMMARY ---")
    print(f"Round-trips detected:      {len(round_trips)}")
    print(f"Layering cases detected:   {len(layering)}")
    print(f"Structuring cases:         {len(structuring)}")
    print(f"Dormant activations:       {len(dormant)}")
    
    critical_count = sum(1 for s in ensemble_scores.values() if s['severity'] == 'CRITICAL')
    high_count = sum(1 for s in ensemble_scores.values() if s['severity'] == 'HIGH')
    print(f"CRITICAL risk nodes:       {critical_count}")
    print(f"HIGH risk nodes:           {high_count}")
    
    # Evaluate against ground truth
    print("\n--- GROUND TRUTH EVALUATION ---")
    predicted_fraud = [n for n, s in ensemble_scores.items() if s['severity'] in ('HIGH', 'CRITICAL')]
    actual_fraud = set(df[df['isFraud'] == 1]['nameOrig'].tolist())
    
    true_positives = [n for n in predicted_fraud if n in actual_fraud]
    false_positives = [n for n in predicted_fraud if n not in actual_fraud]
    
    precision = len(true_positives) / len(predicted_fraud) if predicted_fraud else 0
    recall = len(true_positives) / len(actual_fraud) if actual_fraud else 0
    
    print(f"Precision: {precision:.1%} (Target > 70%)")
    print(f"Recall:    {recall:.1%} (Target > 60%)")
    
    # Peak at one explanation
    if predicted_fraud:
        sample_node = predicted_fraud[0]
        print(f"\nSample Explanation for {sample_node}:\n  {ensemble_scores[sample_node]['explanation']}")
