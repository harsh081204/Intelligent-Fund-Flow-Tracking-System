import networkx as nx
import pandas as pd
import time
import os
import pickle
from data_ingestion import ingest_data_from_parquet

def build_transaction_graph(df):
    """
    Constructs a highly optimized MultiDiGraph from transaction data.
    """
    print("Starting optimized graph construction...")
    start_time = time.time()
    
    # 1. Sort by step first to ensure chronological order for time-dependent patterns
    df = df.sort_values('step').reset_index(drop=True)
    
    # 2. Use MultiDiGraph to preserve all transactions between the same accounts
    G = nx.MultiDiGraph()
    
    # 3. Fast batch edge addition with comprehensive edge attributes
    print("Batch adding edges...")
    edges = list(zip(
        df['nameOrig'],
        df['nameDest'],
        [{'amount': a, 'step': s, 'tx_type': t, 'is_fraud': f, 'balance_before': ob, 'balance_after': nb} 
         for a, s, t, f, ob, nb in zip(
             df['amount'], df['step'], df['type'], df['isFraud'], df['oldbalanceOrg'], df['newbalanceOrig']
         )]
    ))
    G.add_edges_from(edges)
    
    # 4. Compute node stats incrementally without loops by using pandas groupby
    print("Computing node attributes efficiently...")
    
    # Stats as a sender
    sender_stats = df.groupby('nameOrig').agg(
        total_sent=('amount', 'sum'),
        tx_count_out=('amount', 'count'),
        first_seen_out=('step', 'min'),
        last_seen_out=('step', 'max')
    )
    
    # Stats as a receiver
    receiver_stats = df.groupby('nameDest').agg(
        total_received=('amount', 'sum'),
        tx_count_in=('amount', 'count'),
        first_seen_in=('step', 'min'),
        last_seen_in=('step', 'max')
    )
    
    # Node types
    types_out = df.groupby('nameOrig')['type'].apply(set).to_dict()
    types_in = df.groupby('nameDest')['type'].apply(set).to_dict()
    
    # Fraud tracking (if an account was involved in fraud on either side)
    fraud_senders = set(df[df['isFraud'] == 1]['nameOrig'])
    fraud_receivers = set(df[df['isFraud'] == 1]['nameDest'])
    fraud_nodes = fraud_senders.union(fraud_receivers)
    
    # Consolidate attributes for all distinct nodes
    all_nodes = set(df['nameOrig']).union(set(df['nameDest']))
    
    s_dict = sender_stats.to_dict('index')
    r_dict = receiver_stats.to_dict('index')
    
    node_attrs = {}
    for node in all_nodes:
        s = s_dict.get(node, {'total_sent': 0, 'tx_count_out': 0, 'first_seen_out': float('inf'), 'last_seen_out': -1})
        r = r_dict.get(node, {'total_received': 0, 'tx_count_in': 0, 'first_seen_in': float('inf'), 'last_seen_in': -1})
        
        first_seen = min(s['first_seen_out'], r['first_seen_in'])
        last_seen = max(s['last_seen_out'], r['last_seen_in'])
        
        # Combine transaction types and convert to list out of set for safe PyVis serialization
        tx_types = list(types_out.get(node, set()).union(types_in.get(node, set())))
        
        node_attrs[node] = {
            'total_sent': s['total_sent'],
            'total_received': r['total_received'],
            'tx_count': s['tx_count_out'] + r['tx_count_in'],
            'first_seen': first_seen,
            'last_seen': last_seen,
            'tx_types': tx_types,
            'is_fraud': node in fraud_nodes
        }
        
    nx.set_node_attributes(G, node_attrs)
    
    print(f"Graph constructed in {time.time() - start_time:.2f} seconds.")
    return G

def validate_graph(G, df):
    """
    Validates graph health to avoid quiet failures downstream.
    """
    print("\n--- Validating Graph ---")
    print(f"Nodes:          {G.number_of_nodes():,}")
    print(f"Edges:          {G.number_of_edges():,}")
    print(f"Expected edges: {len(df):,}")

    # Check 1: Edge integrity
    assert G.number_of_edges() == len(df), "FAIL: edges lost during construction!"

    # Check 2: Remove self-loops (using keys=True because it's a MultiDiGraph)
    self_loops = list(nx.selfloop_edges(G, keys=True))
    print(f"Self-loops:     {len(self_loops)} (Removed)")
    G.remove_edges_from(self_loops)

    # Check 3: Graph connectivity mapping
    components = nx.number_weakly_connected_components(G)
    print(f"Components:     {components:,}")

    # Check 4: Fraud node presence test
    fraud_df = df[df['isFraud'] == 1]
    if not fraud_df.empty:
        sample_fraud = fraud_df['nameOrig'].iloc[0]
        assert sample_fraud in G.nodes, "FAIL: sample fraud node is missing!"

    print("✅ Graph validation passed\n")
    return G

# ==========================================
# GRAPH CACHING AND INCREMENTAL UPDATES
# ==========================================

def save_graph(G, path="cache/graph.pkl"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(G, f)
    print(f"Graph saved — {G.number_of_nodes()} nodes, "
          f"{G.number_of_edges()} edges")

def load_graph(path="cache/graph.pkl"):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None

def get_graph(df, force_rebuild=False):
    GRAPH_CACHE = "cache/graph.pkl"
    
    # Load from cache if it exists
    if os.path.exists(GRAPH_CACHE) and not force_rebuild:
        print("Loading cached graph...")
        G = load_graph(GRAPH_CACHE)
        if G is not None:
            print(f"Loaded: {G.number_of_nodes():,} nodes, "
                  f"{G.number_of_edges():,} edges")
            return G
    
    # Build fresh
    print("Building graph from scratch...")
    G = build_transaction_graph(df)
    save_graph(G, GRAPH_CACHE)
    return G

def update_graph_with_new_data(G, new_df):
    """
    Add new transactions to existing graph.
    Do NOT rebuild — just extend.
    """
    new_df = new_df.sort_values('step')
    
    for _, row in new_df.iterrows():
        src = row['nameOrig']
        dst = row['nameDest']
        
        # Add nodes if they don't exist yet
        if src not in G:
            G.add_node(src, total_sent=0, total_received=0,
                       tx_count=0, first_seen=row['step'],
                       last_seen=row['step'],
                       tx_types=[], is_fraud=False)
        if dst not in G:
            G.add_node(dst, total_sent=0, total_received=0,
                       tx_count=0, first_seen=row['step'],
                       last_seen=row['step'],
                       tx_types=[], is_fraud=False)
        
        # Add new edge
        G.add_edge(src, dst,
                   amount=row['amount'],
                   step=row['step'],
                   tx_type=row['type'],
                   is_fraud=row['isFraud'],
                   balance_before=row.get('oldbalanceOrg', 0),
                   balance_after=row.get('newbalanceOrig', 0))
        
        # Update node stats incrementally
        G.nodes[src]['total_sent']  += row['amount']
        G.nodes[src]['tx_count']    += 1
        G.nodes[src]['last_seen']    = row['step']
        
        # Ensure tx_types is a list and append if not empty
        if 'tx_types' not in G.nodes[src]: G.nodes[src]['tx_types'] = []
        if row['type'] not in G.nodes[src]['tx_types']: G.nodes[src]['tx_types'].append(row['type'])
        
        G.nodes[dst]['total_received'] += row['amount']
        G.nodes[dst]['tx_count'] += 1
        G.nodes[dst]['last_seen'] = max(row['step'], G.nodes[dst].get('last_seen', 0))
        
        if 'tx_types' not in G.nodes[dst]: G.nodes[dst]['tx_types'] = []
        if row['type'] not in G.nodes[dst]['tx_types']: G.nodes[dst]['tx_types'].append(row['type'])
        
        if row['isFraud']:
            G.nodes[src]['is_fraud'] = True
            G.nodes[dst]['is_fraud'] = True
            
    # Save updated graph
    save_graph(G, "cache/graph.pkl")
    print(f"Graph updated: {G.number_of_nodes():,} nodes, "
          f"{G.number_of_edges():,} edges")
    return G

if __name__ == "__main__":
    try:
        df = ingest_data_from_parquet('sampled_transactions.parquet')
        print(f"Loaded {len(df)} transactions.")
        
        # Separate graph architecture logic cleanly: build -> validate
        G = get_graph(df)
        G = validate_graph(G, df)
        
        print("Ready for pattern detection or ML scoring!")
    except FileNotFoundError:
        print("Error: 'sampled_transactions.parquet' not found in current directory. Please run the data prep first.")