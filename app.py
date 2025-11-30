"""
Interactive Streamlit Web App for Visualizing Contrastive Learning Results
Enhanced with dynamic explanations, guided exploration, and rich interactivity
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
from pathlib import Path
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, v_measure_score, silhouette_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
import time

# Page configuration
st.set_page_config(
    page_title="Contrastive Paper Embeddings",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with improved styling
st.markdown("""
<style>
    /* Dark theme */
    .stApp {
        background: linear-gradient(180deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
    }
    
    /* Animated gradient header */
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #00d4ff 0%, #7c3aed 50%, #f472b6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 0.5rem;
        padding: 1rem;
        animation: shimmer 3s ease-in-out infinite;
    }
    
    @keyframes shimmer {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.8; }
    }
    
    .sub-header {
        font-size: 1.3rem;
        color: #94a3b8;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    
    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d4a6f 100%);
        border-left: 4px solid #00d4ff;
        padding: 1rem 1.5rem;
        border-radius: 0 10px 10px 0;
        margin: 1rem 0;
        color: #e2e8f0;
    }
    
    .insight-box {
        background: linear-gradient(135deg, #1a3d2e 0%, #2d5a47 100%);
        border-left: 4px solid #10b981;
        padding: 1rem 1.5rem;
        border-radius: 0 10px 10px 0;
        margin: 1rem 0;
        color: #e2e8f0;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #4a3728 0%, #5c4433 100%);
        border-left: 4px solid #f59e0b;
        padding: 1rem 1.5rem;
        border-radius: 0 10px 10px 0;
        margin: 1rem 0;
        color: #e2e8f0;
    }
    
    /* Metric explanation */
    .metric-explain {
        font-size: 0.85rem;
        color: #94a3b8;
        margin-top: 0.25rem;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e1e3f 0%, #2d2d5a 100%);
    }
    
    /* Metrics */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%);
        border: 1px solid #3d7ab8;
        border-radius: 12px;
        padding: 1rem;
    }
    
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #00d4ff !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #1a1a2e;
        border-radius: 10px;
        padding: 0.5rem;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #7c3aed 0%, #00d4ff 100%);
        color: white !important;
        border-radius: 8px;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #7c3aed 0%, #00d4ff 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(124, 58, 237, 0.4);
    }
    
    h1, h2, h3 { color: #e2e8f0 !important; }
    p, span, li { color: #cbd5e1; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load the paper dataset."""
    data_path = Path("data/arxiv_papers.parquet")
    if data_path.exists():
        return pd.read_parquet(data_path)
    return None


@st.cache_resource
def load_model():
    """Load the trained model."""
    from model import ContrastivePaperModel
    from config import ModelConfig
    
    checkpoint_path = Path("checkpoints/best_model.pt")
    if not checkpoint_path.exists():
        return None
    
    config = ModelConfig()
    model = ContrastivePaperModel(config)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, checkpoint.get('epoch', 0) + 1, checkpoint.get('loss', 0)


@st.cache_data
def compute_embeddings(_model, texts, batch_size=32):
    """Compute embeddings for texts."""
    return _model.encode_texts(texts, device='cpu', batch_size=batch_size, use_projection=False).numpy()


@st.cache_data
def reduce_dimensions(embeddings, method='tsne', n_components=2, perplexity=30):
    """Reduce embedding dimensions for visualization."""
    if method == 'tsne':
        reducer = TSNE(n_components=n_components, random_state=42, perplexity=min(perplexity, len(embeddings)-1))
    else:
        reducer = PCA(n_components=n_components, random_state=42)
    return reducer.fit_transform(embeddings)


def cluster_and_evaluate(embeddings, true_labels, n_clusters, method='kmeans'):
    """Cluster embeddings and compute metrics."""
    if method == 'kmeans':
        clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    elif method == 'spectral':
        clusterer = SpectralClustering(n_clusters=n_clusters, random_state=42, affinity='nearest_neighbors', n_neighbors=10)
    else:
        clusterer = AgglomerativeClustering(n_clusters=n_clusters)
    
    pred_labels = clusterer.fit_predict(embeddings)
    
    metrics = {
        'NMI': normalized_mutual_info_score(true_labels, pred_labels),
        'ARI': adjusted_rand_score(true_labels, pred_labels),
        'V-measure': v_measure_score(true_labels, pred_labels),
    }
    
    if len(np.unique(pred_labels)) > 1:
        metrics['Silhouette'] = silhouette_score(embeddings, pred_labels)
    
    return pred_labels, metrics


def plot_styled_chart(fig, title="", height=500):
    """Apply consistent dark styling to plotly figures."""
    fig.update_layout(
        height=height,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(26,26,46,0.8)',
        font=dict(color='#e2e8f0'),
        title=dict(text=title, font=dict(color='#00d4ff', size=18)),
        legend=dict(font=dict(color='white'), bgcolor='rgba(0,0,0,0.3)'),
        hoverlabel=dict(bgcolor='#1e3a5f', font_size=14, font_color='white')
    )
    fig.update_xaxes(gridcolor='#3d3d6b', zerolinecolor='#3d3d6b')
    fig.update_yaxes(gridcolor='#3d3d6b', zerolinecolor='#3d3d6b')
    return fig


def render_metric_with_explanation(col, label, value, explanation, good_threshold=None):
    """Render a metric with explanation tooltip."""
    col.metric(label, f"{value:.3f}")
    
    if good_threshold:
        if value >= good_threshold:
            color, status = "#10b981", "✅ Good"
        elif value >= good_threshold * 0.7:
            color, status = "#f59e0b", "⚠️ Fair"
        else:
            color, status = "#ef4444", "❌ Poor"
        col.markdown(f"<span style='color:{color}'>{status}</span>", unsafe_allow_html=True)
    
    col.markdown(f"<p class='metric-explain'>{explanation}</p>", unsafe_allow_html=True)


def main():
    # Header
    st.markdown('<p class="main-header">📚 Contrastive Learning for Scientific Papers</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Explore how AI learns to understand research papers without labels</p>', unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    model_data = load_model()
    
    if df is None:
        st.error("❌ No data found. Run `python main.py --mode collect` first.")
        return
    if model_data is None:
        st.error("❌ No model found. Run `python main.py --mode train` first.")
        return
    
    model, best_epoch, best_loss = model_data
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🎯 Quick Guide")
        
        with st.expander("❓ What is this?", expanded=True):
            st.markdown("""
            This app visualizes how **contrastive learning** teaches AI to understand scientific papers.
            
            **The Goal:** Group similar papers together without being told what "similar" means!
            
            **How it works:**
            1. 📄 Take a paper's title + abstract
            2. 🔀 Create two "augmented" versions
            3. 🧠 Train model to recognize they're the same paper
            4. 📊 Result: Papers on similar topics cluster together!
            """)
        
        st.markdown("---")
        st.markdown("### 📊 Dataset Stats")
        
        col1, col2 = st.columns(2)
        col1.metric("Papers", f"{len(df):,}")
        col2.metric("Categories", df['primary_category'].nunique())
        
        col1, col2 = st.columns(2)
        col1.metric("Best Epoch", best_epoch)
        col2.metric("Loss", f"{best_loss:.4f}")
        
        st.markdown("---")
        st.markdown("### ⚙️ Controls")
        
        max_samples = st.slider("Sample Size", 100, min(2000, len(df)), min(500, len(df)), 100,
                                help="More samples = slower but more accurate visualization")
        
        dim_reduction = st.radio("Dimension Reduction", ["t-SNE", "PCA"],
                                 help="t-SNE: Better for clusters | PCA: Faster")
        
        perplexity = st.slider("Perplexity", 5, 50, 30, help="Higher = more global structure") if dim_reduction == "t-SNE" else 30
    
    # Sample data
    df_sample = df.sample(n=max_samples, random_state=42) if len(df) > max_samples else df
    texts = (df_sample['title'] + ". " + df_sample['abstract']).tolist()
    categories = df_sample['primary_category'].tolist()
    
    le = LabelEncoder()
    true_labels = le.fit_transform(categories)
    n_clusters = len(le.classes_)
    category_names = {i: cat for i, cat in enumerate(le.classes_)}
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔍 Explore Embeddings", "📈 Understand Metrics", "🏆 Compare Methods",
        "📝 Browse Papers", "🧪 Try Augmentations"
    ])
    
    # TAB 1: EXPLORE EMBEDDINGS
    with tab1:
        st.markdown("""
        <div class='info-box'>
        <strong>🎯 What you're looking at:</strong> Each dot is a scientific paper. 
        Papers that the model thinks are similar appear close together. 
        Colors show actual categories - same-colored dots should cluster if the model learned well!
        </div>
        """, unsafe_allow_html=True)
        
        with st.spinner("🧠 Computing paper embeddings..."):
            embeddings = compute_embeddings(model, texts)
        
        with st.spinner(f"📐 Reducing to 2D with {dim_reduction}..."):
            emb_2d = reduce_dimensions(embeddings, 'tsne' if dim_reduction == "t-SNE" else 'pca', perplexity=perplexity)
        
        # Controls
        st.markdown("### 🎮 Interactive Controls")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            color_by = st.selectbox("Color points by:", ["True Category", "Predicted Cluster"],
                                    help="Change what the colors represent")
        with col2:
            clustering_method = st.selectbox("Clustering Algorithm:", ["K-means", "Spectral", "Agglomerative"],
                                             help="K-means: Fast | Spectral: Complex shapes | Agglomerative: Hierarchical")
        with col3:
            highlight_category = st.selectbox("Highlight Category:", ["None"] + list(category_names.values()),
                                              help="Make one category stand out")
        
        # Compute clusters
        cluster_map = {"K-means": "kmeans", "Spectral": "spectral", "Agglomerative": "agglomerative"}
        pred_labels, metrics = cluster_and_evaluate(embeddings, true_labels, n_clusters, cluster_map[clustering_method])
        
        # Create plot
        hover_template = "<b>%{customdata[0]}</b><br><i>Category: %{customdata[1]}</i><br>Cluster: %{customdata[2]}<extra></extra>"
        
        fig = go.Figure()
        
        if highlight_category != "None":
            mask = np.array([c == highlight_category for c in categories])
            
            # Background
            fig.add_trace(go.Scatter(
                x=emb_2d[~mask, 0], y=emb_2d[~mask, 1], mode='markers',
                marker=dict(size=8, color='#3d3d6b', opacity=0.3), name='Other Papers',
                customdata=[[df_sample.iloc[i]['title'][:50]+"...", categories[i], pred_labels[i]] for i in np.where(~mask)[0]],
                hovertemplate=hover_template
            ))
            
            # Highlighted
            fig.add_trace(go.Scatter(
                x=emb_2d[mask, 0], y=emb_2d[mask, 1], mode='markers',
                marker=dict(size=14, color='#00d4ff', opacity=1, line=dict(width=2, color='white'), symbol='star'),
                name=highlight_category,
                customdata=[[df_sample.iloc[i]['title'][:50]+"...", categories[i], pred_labels[i]] for i in np.where(mask)[0]],
                hovertemplate=hover_template
            ))
        else:
            colors = [category_names[l] for l in true_labels] if color_by == "True Category" else [f"Cluster {l}" for l in pred_labels]
            
            for idx, cat in enumerate(le.classes_ if color_by == "True Category" else [f"Cluster {i}" for i in range(n_clusters)]):
                if color_by == "True Category":
                    mask = np.array(categories) == cat
                else:
                    mask = pred_labels == idx
                
                fig.add_trace(go.Scatter(
                    x=emb_2d[mask, 0], y=emb_2d[mask, 1], mode='markers',
                    marker=dict(size=10, opacity=0.8, line=dict(width=1, color='white')),
                    name=cat,
                    customdata=[[df_sample.iloc[i]['title'][:50]+"...", categories[i], pred_labels[i]] for i in np.where(mask)[0]],
                    hovertemplate=hover_template
                ))
        
        fig = plot_styled_chart(fig, f"Paper Embeddings ({dim_reduction}) - Hover for details!", height=600)
        fig.update_layout(xaxis_title="Dimension 1", yaxis_title="Dimension 2",
                          legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02))
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Interpretation
        st.markdown("### 🔎 How to Interpret This")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class='insight-box'>
            <strong>✅ Good signs:</strong>
            <ul>
                <li>Same-colored points form tight clusters</li>
                <li>Different colors are well-separated</li>
                <li>Clear structure visible in the space</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class='warning-box'>
            <strong>⚠️ What to watch for:</strong>
            <ul>
                <li>Colors heavily mixed = poor separation</li>
                <li>Random scatter = model didn't learn</li>
                <li>Some overlap is normal (papers span topics!)</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Metrics
        st.markdown("### 📊 Clustering Quality Scores")
        cols = st.columns(4)
        
        render_metric_with_explanation(cols[0], "NMI", metrics['NMI'], "How well clusters match categories (0-1)", 0.3)
        render_metric_with_explanation(cols[1], "ARI", metrics['ARI'], "Agreement beyond random chance (-1 to 1)", 0.2)
        render_metric_with_explanation(cols[2], "V-measure", metrics['V-measure'], "Balance of homogeneity & completeness", 0.3)
        if 'Silhouette' in metrics:
            render_metric_with_explanation(cols[3], "Silhouette", metrics['Silhouette'], "How distinct clusters are (-1 to 1)", 0.2)
    
    # TAB 2: UNDERSTAND METRICS
    with tab2:
        st.markdown("## 📈 Understanding the Metrics")
        
        st.markdown("""
        <div class='info-box'>
        <strong>Why metrics matter:</strong> We need numbers to objectively measure 
        how well our model learned to group similar papers.
        </div>
        """, unsafe_allow_html=True)
        
        metric_choice = st.selectbox("Select a metric to explore:",
                                     ["NMI (Normalized Mutual Information)", "ARI (Adjusted Rand Index)",
                                      "V-measure", "Silhouette Score"])
        
        if "NMI" in metric_choice:
            st.markdown("""
            ### 📊 NMI: Normalized Mutual Information
            
            **What it measures:** How much knowing the cluster tells you about the true category.
            
            | Score | Meaning |
            |-------|---------|
            | 0.0 | Clusters are random |
            | 0.3 | Moderate - clusters partially capture categories |
            | 0.5+ | Good - clusters strongly relate to categories |
            | 1.0 | Perfect - each cluster = one category |
            """)
            
            # Interactive demo
            st.markdown("#### 🎮 Interactive Example")
            demo_nmi = st.slider("Adjust cluster quality:", 0.0, 1.0, 0.5, 0.1)
            
            np.random.seed(42)
            n_demo = 100
            
            if demo_nmi > 0.7:
                demo_data = np.vstack([np.random.randn(34, 2) + [0, 3], np.random.randn(33, 2) + [3, 0], np.random.randn(33, 2) + [-3, 0]])
                demo_colors = ['Category A']*34 + ['Category B']*33 + ['Category C']*33
            elif demo_nmi > 0.3:
                demo_data = np.vstack([np.random.randn(34, 2) * 1.5 + [0, 2], np.random.randn(33, 2) * 1.5 + [2, 0], np.random.randn(33, 2) * 1.5 + [-2, 0]])
                demo_colors = ['Category A']*34 + ['Category B']*33 + ['Category C']*33
            else:
                demo_data = np.random.randn(n_demo, 2) * 3
                demo_colors = list(np.random.choice(['Category A', 'Category B', 'Category C'], n_demo))
            
            fig_demo = px.scatter(x=demo_data[:, 0], y=demo_data[:, 1], color=demo_colors,
                                  color_discrete_sequence=['#00d4ff', '#f472b6', '#10b981'])
            fig_demo = plot_styled_chart(fig_demo, f"Example with NMI ≈ {demo_nmi:.1f}", height=350)
            st.plotly_chart(fig_demo, use_container_width=True)
            
        elif "ARI" in metric_choice:
            st.markdown("""
            ### 📊 ARI: Adjusted Rand Index
            
            **What it measures:** Agreement between clusterings, adjusted for chance.
            
            | Score | Meaning |
            |-------|---------|
            | < 0 | Worse than random |
            | 0.0 | Random clustering |
            | 0.5 | Good agreement |
            | 1.0 | Perfect agreement |
            """)
            
        elif "V-measure" in metric_choice:
            st.markdown("""
            ### 📊 V-measure
            
            **What it measures:** The harmonic mean of:
            - **Homogeneity:** Are cluster members from the same category?
            - **Completeness:** Are category members in the same cluster?
            """)
            
            from sklearn.metrics import homogeneity_score, completeness_score
            h, c = homogeneity_score(true_labels, pred_labels), completeness_score(true_labels, pred_labels)
            
            fig_hc = go.Figure(go.Bar(x=['Homogeneity', 'Completeness', 'V-measure'], y=[h, c, metrics['V-measure']],
                                      marker_color=['#00d4ff', '#f472b6', '#10b981'],
                                      text=[f'{v:.3f}' for v in [h, c, metrics['V-measure']]], textposition='auto'))
            fig_hc = plot_styled_chart(fig_hc, "Your Model's Scores", height=350)
            st.plotly_chart(fig_hc, use_container_width=True)
        else:
            st.markdown("""
            ### 📊 Silhouette Score
            
            **What it measures:** How similar points are to their own cluster vs. other clusters.
            
            | Score | Meaning |
            |-------|---------|
            | < 0 | Points in wrong clusters |
            | 0.0 | Clusters overlap |
            | 0.5 | Clear structure |
            | 1.0 | Perfect clusters |
            """)
        
        # Compare methods
        st.markdown("### 🔄 Compare All Clustering Methods")
        
        all_methods = {name: cluster_and_evaluate(embeddings, true_labels, n_clusters, key)[1]
                       for name, key in [("K-means", "kmeans"), ("Spectral", "spectral"), ("Agglomerative", "agglomerative")]}
        
        fig_compare = go.Figure()
        for i, metric_name in enumerate(['NMI', 'ARI', 'V-measure']):
            fig_compare.add_trace(go.Bar(
                name=metric_name, x=list(all_methods.keys()),
                y=[all_methods[m][metric_name] for m in all_methods.keys()],
                text=[f"{all_methods[m][metric_name]:.3f}" for m in all_methods.keys()],
                textposition='auto', marker_color=['#00d4ff', '#7c3aed', '#f472b6'][i]
            ))
        
        fig_compare = plot_styled_chart(fig_compare, "Clustering Methods Comparison", height=400)
        fig_compare.update_layout(barmode='group')
        st.plotly_chart(fig_compare, use_container_width=True)
        
        best_method = max(all_methods.keys(), key=lambda x: all_methods[x]['NMI'])
        st.markdown(f"""
        <div class='insight-box'>
        <strong>💡 Recommendation:</strong> Based on NMI, <strong>{best_method}</strong> works best 
        for this data (NMI = {all_methods[best_method]['NMI']:.3f}).
        </div>
        """, unsafe_allow_html=True)
    
    # TAB 3: COMPARE METHODS
    with tab3:
        st.markdown("## 🏆 How Does Our Model Compare?")
        
        st.markdown("""
        <div class='info-box'>
        <strong>The question:</strong> Did contrastive learning actually help? 
        Let's compare against baselines that don't require training.
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander("📚 What are the baselines?", expanded=True):
            col1, col2, col3 = st.columns(3)
            col1.markdown("**🔵 TF-IDF**\n- Counts word frequencies\n- No understanding of meaning\n- Very fast")
            col2.markdown("**🟡 Pre-trained Transformer**\n- Trained on 1B+ sentences\n- Understands language\n- No fine-tuning")
            col3.markdown("**🟢 Contrastive (Ours)**\n- Learned from ArXiv\n- Adapted to scientific text\n- ~5K papers, 30 epochs")
        
        results_df = pd.DataFrame({
            'Method': ['Contrastive (Ours)', 'TF-IDF', 'Pre-trained ST'],
            'K-means': [0.324, 0.281, 0.361], 'Spectral': [0.344, 0.271, 0.386], 'Agglomerative': [0.292, 0.171, 0.284]
        })
        
        colors = {'Contrastive (Ours)': '#10b981', 'TF-IDF': '#ef4444', 'Pre-trained ST': '#3b82f6'}
        
        fig = make_subplots(rows=1, cols=3, subplot_titles=['K-means', 'Spectral', 'Agglomerative'])
        
        for i, cluster_method in enumerate(['K-means', 'Spectral', 'Agglomerative']):
            for method in results_df['Method']:
                val = results_df[results_df['Method'] == method][cluster_method].values[0]
                fig.add_trace(go.Bar(x=[method], y=[val], name=method if i == 0 else None,
                                     marker_color=colors[method], text=[f'{val:.3f}'], textposition='auto',
                                     showlegend=(i == 0)), row=1, col=i+1)
        
        fig = plot_styled_chart(fig, "", height=400)
        fig.update_layout(barmode='group')
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div class='insight-box'>
            <strong>✅ We beat TF-IDF everywhere!</strong>
            <ul><li>K-means: +15%</li><li>Spectral: +27%</li><li>Agglomerative: +71%</li></ul>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class='warning-box'>
            <strong>⚡ Pre-trained ST is strong</strong>
            <ul><li>Trained on 1B+ pairs</li><li>We used only 5K papers</li><li>Yet we match it on Agglomerative!</li></ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Radar chart
        fig_radar = go.Figure()
        color_map = {
            'Contrastive (Ours)': ('#10b981', 'rgba(16, 185, 129, 0.2)'),
            'TF-IDF': ('#ef4444', 'rgba(239, 68, 68, 0.2)'),
            'Pre-trained ST': ('#3b82f6', 'rgba(59, 130, 246, 0.2)')
        }
        
        for method, (line_color, fill_color) in color_map.items():
            row = results_df[results_df['Method'] == method].iloc[0]
            fig_radar.add_trace(go.Scatterpolar(
                r=[row['K-means'], row['Spectral'], row['Agglomerative'], row['K-means']],
                theta=['K-means', 'Spectral', 'Agglomerative', 'K-means'],
                fill='toself', name=method, line=dict(color=line_color, width=3), fillcolor=fill_color
            ))
        
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 0.5], gridcolor='#3d3d6b'),
                       angularaxis=dict(gridcolor='#3d3d6b'), bgcolor='rgba(26,26,46,0.8)'),
            paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#e2e8f0'), height=450, legend=dict(font=dict(color='white'))
        )
        st.plotly_chart(fig_radar, use_container_width=True)
    
    # TAB 4: BROWSE PAPERS
    with tab4:
        st.markdown("## 📝 Explore the Papers")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### 📂 Browse by Category")
            selected_cat = st.selectbox("Choose category:", list(category_names.values()))
            cat_df = df_sample[df_sample['primary_category'] == selected_cat]
            st.metric("Papers in category", len(cat_df))
        
        with col2:
            st.markdown("### 📄 Papers")
            for i, (_, row) in enumerate(cat_df.head(5).iterrows()):
                with st.expander(f"📄 {row['title'][:70]}...", expanded=(i == 0)):
                    st.markdown(f"**{row['title']}**")
                    st.markdown(f"*Category: `{row['primary_category']}`*")
                    st.markdown(row['abstract'][:400] + "...")
        
        st.markdown("---")
        st.markdown("### 🔍 Find Similar Papers")
        
        paper_idx = st.selectbox("Select a paper:", range(len(df_sample)),
                                 format_func=lambda x: f"{df_sample.iloc[x]['title'][:60]}...")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📄 Selected Paper")
            selected = df_sample.iloc[paper_idx]
            st.markdown(f"**{selected['title']}**")
            st.markdown(f"*Category: `{selected['primary_category']}`*")
            st.markdown(selected['abstract'][:300] + "...")
        
        with col2:
            if st.button("🔍 Find Similar Papers", use_container_width=True):
                query_emb = embeddings[paper_idx]
                similarities = np.dot(embeddings, query_emb) / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_emb) + 1e-8)
                top_indices = np.argsort(similarities)[::-1][1:6]
                
                st.markdown("#### 🎯 Most Similar Papers")
                for rank, idx in enumerate(top_indices, 1):
                    row = df_sample.iloc[idx]
                    same_cat = row['primary_category'] == selected['primary_category']
                    st.markdown(f"**{rank}. {'✅' if same_cat else '🔄'} {row['title'][:50]}...**\n\n"
                               f"Category: `{row['primary_category']}` | Similarity: `{similarities[idx]:.3f}`")
                
                same_count = sum(1 for idx in top_indices if df_sample.iloc[idx]['primary_category'] == selected['primary_category'])
                if same_count >= 3:
                    st.success(f"✅ {same_count}/5 similar papers are from the same category!")
                else:
                    st.info(f"🔄 {same_count}/5 from same category. Papers often span topics!")
    
    # TAB 5: AUGMENTATIONS
    with tab5:
        st.markdown("## 🧪 Explore Text Augmentations")
        
        st.markdown("""
        <div class='info-box'>
        <strong>The secret sauce:</strong> Contrastive learning needs "positive pairs" - 
        two versions of the same paper that should have similar embeddings.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 🔄 How Contrastive Learning Works")
        
        col1, col2, col3 = st.columns(3)
        col1.markdown("<div style='text-align:center;padding:1rem;background:#1e3a5f;border-radius:10px;'>"
                     "<h4 style='color:#00d4ff;'>1️⃣ Original</h4>"
                     "<p>\"Deep learning improves NLP significantly.\"</p></div>", unsafe_allow_html=True)
        col2.markdown("<div style='text-align:center;padding:1rem;background:#1a3d2e;border-radius:10px;'>"
                     "<h4 style='color:#10b981;'>2️⃣ Augment</h4>"
                     "<p>View 1: \"Deep learning improves...\"<br>View 2: \"Neural networks enhance...\"</p></div>", unsafe_allow_html=True)
        col3.markdown("<div style='text-align:center;padding:1rem;background:#3d2d5a;border-radius:10px;'>"
                     "<h4 style='color:#7c3aed;'>3️⃣ Train</h4>"
                     "<p>Model learns: View 1 ≈ View 2</p></div>", unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 🎮 Try It Yourself!")
        
        sample_text = st.text_area("Enter text to augment:",
            value="Deep learning has transformed natural language processing. "
                  "Transformer models achieve state-of-the-art results. "
                  "Attention mechanisms capture long-range dependencies effectively.", height=100)
        
        from augmentations import WordDropout, WordShuffle, SentenceShuffle, SynonymReplacement, SpanMasking, RandomInsertion
        
        col1, col2, col3 = st.columns(3)
        with col1:
            use_dropout = st.checkbox("🗑️ Word Dropout", value=True)
            dropout_prob = st.slider("Dropout %", 5, 30, 10, disabled=not use_dropout) / 100
        with col2:
            use_synonym = st.checkbox("📖 Synonym Replace", value=True)
            synonym_prob = st.slider("Replace %", 5, 30, 15, disabled=not use_synonym) / 100
        with col3:
            use_mask = st.checkbox("🎭 Span Masking")
            use_shuffle = st.checkbox("🔀 Word Shuffle")
        
        if st.button("🔄 Generate Augmented Views", use_container_width=True):
            col1, col2 = st.columns(2)
            
            result1, result2 = sample_text, sample_text
            
            if use_dropout:
                result1 = WordDropout(dropout_prob)(result1)
                result2 = WordDropout(dropout_prob * 1.5)(result2)
            if use_synonym:
                result1 = SynonymReplacement(synonym_prob)(result1)
                result2 = SynonymReplacement(synonym_prob * 1.2)(result2)
            if use_mask:
                result2 = SpanMasking(0.15)(result2)
            if use_shuffle:
                result1 = WordShuffle(3)(result1)
                result2 = WordShuffle(3)(result2)
            
            with col1:
                st.markdown("#### View 1 (Lighter)")
                st.success(result1)
            with col2:
                st.markdown("#### View 2 (Stronger)")
                st.info(result2)
        
        # Similarity matrix
        st.markdown("### 📊 Training Visualization")
        
        batch_size = 4
        np.random.seed(42)
        
        sim_matrix = np.random.rand(batch_size * 2, batch_size * 2) * 0.3
        for i in range(batch_size):
            sim_matrix[i, i + batch_size] = 0.85 + np.random.rand() * 0.1
            sim_matrix[i + batch_size, i] = 0.85 + np.random.rand() * 0.1
        np.fill_diagonal(sim_matrix, 1.0)
        
        labels = [f"Paper{i}_v1" for i in range(batch_size)] + [f"Paper{i}_v2" for i in range(batch_size)]
        
        fig_sim = px.imshow(sim_matrix, x=labels, y=labels, color_continuous_scale="RdYlGn", labels=dict(color="Similarity"))
        
        for i in range(batch_size):
            fig_sim.add_annotation(x=i + batch_size, y=i, text="✓", font=dict(size=20, color="white"), showarrow=False)
            fig_sim.add_annotation(x=i, y=i + batch_size, text="✓", font=dict(size=20, color="white"), showarrow=False)
        
        fig_sim = plot_styled_chart(fig_sim, "Similarity Matrix (✓ = Positive Pairs)", height=450)
        st.plotly_chart(fig_sim, use_container_width=True)
        
        st.markdown("""
        <div class='insight-box'>
        <strong>The key insight:</strong> By learning that augmented versions of the same paper 
        should be similar, the model discovers that papers about similar topics should also 
        be similar - without ever being told what the topics are!
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
