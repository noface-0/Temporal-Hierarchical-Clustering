import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import linear_sum_assignment
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import networkx as nx
import datetime
import os
import warnings
warnings.filterwarnings('ignore')

class TemporalMarketClustering:
    def __init__(self, 
                start_date='2016-01-01', 
                end_date='2025-03-20',
                window_size=90,
                step_size=30,
                max_clusters=8,
                n_tickers=40,
                show_all_scores=False,
                metric='davies_bouldin',
                linkage_method='average'):
        """
        Initialize the temporal clustering analysis.
        
        Parameters:
        -----------
        start_date : str
            Start date for analysis in 'YYYY-MM-DD' format
        end_date : str
            End date for analysis in 'YYYY-MM-DD' format
        window_size : int
            Size of each time window in days
        step_size : int
            Number of days to move between consecutive windows
        max_clusters : int
            Maximum number of clusters to consider.
            Can be set up to n_tickers, but will be limited internally based on data size.
        n_tickers : int
            Number of tickers to use in the analysis
        show_all_scores : bool
            If True, print cluster quality scores for all tested cluster counts
        metric : str
            Clustering quality metric to use ('silhouette', 'davies_bouldin', or 'calinski_harabasz')
        linkage_method : str
            Linkage method for hierarchical clustering ('ward', 'average', 'complete', or 'single')
        """
        self.start_date = start_date
        self.end_date = end_date
        self.window_size = window_size
        self.step_size = step_size
        self.max_clusters = max_clusters
        self.n_tickers = n_tickers
        self.show_all_scores = show_all_scores
        self.metric = metric
        self.linkage_method = linkage_method
        
        # Results will be stored here
        self.data = None
        self.results = None
        self.time_windows = None
        self.cluster_graph = None
        
    def get_sp500_tickers(self):
        """Get a subset of S&P 500 tickers and popular ETFs for analysis."""
        # Extended list of stocks and ETFs for analysis
        sample_tickers = [
            # Technology
            'AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'ADBE', 'CRM', 'AVGO', 'INTC', 'AMD', 
            'CSCO', 'ORCL', 'IBM', 'TXN', 'QCOM', 'NFLX', 'PYPL', 'UBER', 'SNOW', 'PLTR',
            # Financial
            'JPM', 'BAC', 'WFC', 'GS', 'MS', 'V', 'MA', 'AXP', 'C', 'BLK',
            # Healthcare
            'JNJ', 'UNH', 'PFE', 'ABBV', 'MRK', 'ABT', 'TMO', 'LLY', 'AMGN', 'BMY',
            # Consumer
            'AMZN', 'TSLA', 'WMT', 'PG', 'HD', 'COST', 'MCD', 'NKE', 'SBUX', 'DIS',
            # Energy & Industrial
            'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'HON', 'UNP', 'GE', 'CAT', 'DE',
            # Communication & Media
            'CMCSA', 'VZ', 'T', 'TMUS', 'CHTR', 'FOX', 'NWSA', 'OMC', 'IPG',
            # Utilities & Real Estate
            'NEE', 'DUK', 'SO', 'D', 'AEP', 'AMT', 'PLD', 'SPG', 'WELL', 'EQIX',
            # ETFs - Market Index
            'SPY', 'QQQ', 'DIA', 'IWM', 'VTI', 'VOO', 'VEA', 'VWO', 'EFA', 'EEM',
            # ETFs - Sectors
            'XLK', 'XLF', 'XLV', 'XLP', 'XLY', 'XLE', 'XLI', 'XLU', 'XLB', 'XLRE',
            # ETFs - Fixed Income
            'AGG', 'BND', 'TLT', 'IEF', 'LQD', 'HYG', 'MUB', 'TIP', 'SHY', 'VCSH',
            # ETFs - Commodities & Others
            'GLD', 'SLV', 'USO', 'UNG', 'DBC', 'VNQ', 'REET', 'VXX', 'UVXY', 'SVXY',
            # ETFs - Factor & Strategy
            'MTUM', 'QUAL', 'SIZE', 'VLUE', 'USMV', 'HDV', 'VIG', 'NOBL', 'RSP', 'SDY',
            # ETFs - ARK Funds
            'ARKK', 'ARKW', 'ARKG', 'ARKF', 'ARKX', 'PRNT', 'IZRL'
        ]
        return sample_tickers[:self.n_tickers]
    
    def download_data(self):
        """Download historical price data for the selected tickers."""
        tickers = self.get_sp500_tickers()
        print(f"Downloading data for {len(tickers)} tickers...")
        
        # Get all data and then select Close prices
        raw_data = yf.download(tickers, start=self.start_date, end=self.end_date)
        print("Available columns:", raw_data.columns.tolist())
        
        # Try to get Adj Close if available, otherwise use Close
        if 'Adj Close' in raw_data.columns:
            data = raw_data['Adj Close']
        elif 'Close' in raw_data.columns:
            data = raw_data['Close']
        else:
            # If the structure is different, try to get the first level of columns
            try:
                data = raw_data.iloc[:, 0].to_frame()
                print("Using first available column as data")
            except:
                print("Couldn't extract price data from the downloaded information")
                return pd.DataFrame()
                
        # Drop any tickers with missing data
        data = data.dropna(axis=1)
        print(f"Successfully downloaded data for {data.shape[1]} tickers")
        
        self.data = data
        return data
    
    def calculate_features(self, window):
        # Calculate daily returns
        returns = self.data.pct_change().dropna()
        
        # Extract returns for the specific window
        window_returns = returns.iloc[window[0]:window[1]]
        
        # Calculate features
        features = pd.DataFrame(index=returns.columns)
        
        # Basic statistical features
        features['mean_return'] = window_returns.mean()
        features['volatility'] = window_returns.std()
        features['downside_risk'] = window_returns[window_returns < 0].std()
        
        # Maximum drawdown
        cum_returns = (1 + window_returns).cumprod()
        running_max = cum_returns.cummax()
        drawdown = (cum_returns / running_max) - 1
        features['max_drawdown'] = drawdown.min()
        
        # Calculate correlation matrix
        corr_matrix = window_returns.corr()
        
        # Standardize features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        
        return pd.DataFrame(scaled_features, index=features.index, columns=features.columns), corr_matrix
    
    def perform_clustering(self, features, corr_matrix, n_clusters):
        # Use correlation distance (1 - correlation)
        distance_matrix = 1 - corr_matrix
        
        # Convert to condensed form for linkage function
        condensed_dist = pdist(distance_matrix)
        
        # Perform hierarchical clustering with the specified linkage method
        Z = linkage(condensed_dist, method=self.linkage_method)
        
        # Get cluster labels
        labels = fcluster(Z, n_clusters, criterion='maxclust')
        
        # Calculate quality score based on selected metric
        if len(np.unique(labels)) > 1:  # Need at least 2 clusters for metrics
            if self.metric == 'silhouette':
                score = silhouette_score(distance_matrix, labels)
            elif self.metric == 'davies_bouldin':
                # For Davies-Bouldin, lower is better, so we negate for consistency
                # We use features here rather than distance matrix for DB index
                score = -davies_bouldin_score(features, labels)
            elif self.metric == 'calinski_harabasz':
                score = calinski_harabasz_score(features, labels)
            else:
                # Default to silhouette if an unknown metric is specified
                score = silhouette_score(distance_matrix, labels)
        else:
            score = 0
        
        return labels, Z, score
    
    def create_tsne(self, corr_matrix, labels):
        # Use correlation distance
        distance_matrix = 1 - corr_matrix
        
        n_samples = distance_matrix.shape[0]
        
        # For small datasets, use MDS instead of t-SNE
        if n_samples < 10:
            print(f"  Using MDS instead of t-SNE for small dataset (n={n_samples})")
            from sklearn.manifold import MDS
            mds = MDS(n_components=2, random_state=42, dissimilarity='precomputed')
            return mds.fit_transform(distance_matrix)
        
        # Apply t-SNE for dimensionality reduction with adjusted perplexity
        perplexity = min(30, n_samples - 1)  # Default is 30, but must be < n_samples
        tsne = TSNE(n_components=2, random_state=42, metric='precomputed', 
                   perplexity=perplexity, init='random')  # Use random init with precomputed
        tsne_results = tsne.fit_transform(distance_matrix)
        
        return tsne_results
    
    def get_optimal_clusters(self, features, corr_matrix):
        cluster_scores = []
        n_samples = len(features)
        
        # Calculate theoretical max clusters (can't have more clusters than samples)
        # But we also need at least 2 samples per cluster for most metrics to work
        theoretical_max = max(2, n_samples // 2)
        
        # Allow user to set max_clusters up to the number of equities, but cap it internally
        # to ensure we don't attempt invalid cluster numbers
        effective_max = min(self.max_clusters, theoretical_max, n_samples - 1)
        
        # If max_clusters is very high relative to sample size, print information
        if self.max_clusters > theoretical_max:
            print(f"  Note: max_clusters ({self.max_clusters}) exceeds theoretical maximum ({theoretical_max})")
            print(f"  Testing cluster counts from 2 to {effective_max}")
        
        # Try different numbers of clusters and calculate the quality score
        scores_by_n = {}  # Store scores by cluster count for reporting
        
        for n_clusters in range(2, effective_max + 1):
            labels, _, score = self.perform_clustering(features, corr_matrix, n_clusters)
            cluster_scores.append(score)
            scores_by_n[n_clusters] = score
        
        # If requested, show scores for all cluster counts
        if hasattr(self, 'show_all_scores') and self.show_all_scores:
            print(f"  {self.metric.capitalize()} scores by cluster count:")
            for n, score in sorted(scores_by_n.items()):
                print(f"  {n} clusters: {score:.3f}")
        
        # Get the number of clusters with the best score (highest for most metrics)
        if cluster_scores:
            # For Davies-Bouldin, we've already negated in perform_clustering, so argmax works
            optimal_clusters = np.argmax(cluster_scores) + 2  # +2 because we start from 2 clusters
            
            metric_name = self.metric.replace('_', ' ').capitalize()
            print(f"  Selected {optimal_clusters} clusters with {metric_name} score: {scores_by_n[optimal_clusters]:.3f}")
            
            # For Davies-Bouldin, show the actual (non-negated) score for clarity
            if self.metric == 'davies_bouldin':
                actual_score = -scores_by_n[optimal_clusters]
                print(f"  (Lower Davies-Bouldin score is better, actual value: {actual_score:.3f})")
        else:
            optimal_clusters = 2  # Default to 2 clusters if we couldn't calculate scores
            
        return optimal_clusters
    
    def run_clustering(self):
        if self.data is None:
            self.download_data()
            
        if self.data.empty:
            print("No data available for clustering.")
            return [], []
        
        time_windows = []
        n_days = len(self.data)
        
        # Create time windows
        for start in range(0, n_days - self.window_size, self.step_size):
            end = start + self.window_size
            time_windows.append((start, end))
        
        # Store results for each window
        results = []
        
        # Process each time window
        for i, window in enumerate(time_windows):
            # Calculate start and end dates for this window
            start_date = self.data.index[window[0]]
            end_date = self.data.index[window[1]-1]
            print(f"Processing window {i+1}/{len(time_windows)}: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
            
            # Calculate features
            features, corr_matrix = self.calculate_features(window)
            
            # Get optimal number of clusters
            n_clusters = self.get_optimal_clusters(features, corr_matrix)
            print(f"  Optimal clusters: {n_clusters}")
            
            # Perform clustering
            labels, Z, score = self.perform_clustering(features, corr_matrix, n_clusters)
            
            # Create TSNE projection
            tsne_results = self.create_tsne(corr_matrix, labels)
            
            # Store results
            results.append({
                'window': (start_date, end_date),
                'features': features,
                'corr_matrix': corr_matrix,
                'n_clusters': n_clusters,
                'labels': labels,
                'linkage': Z,
                'tsne': tsne_results,
                'tickers': self.data.columns,
                'silhouette': score
            })
        
        self.results = results
        self.time_windows = time_windows
        
        return results, time_windows
    
    def track_clusters(self):
        if self.results is None:
            self.run_clustering()
            
        if not self.results:
            print("No clustering results to track.")
            return []
        
        results = self.results
        n_windows = len(results)
        
        # Store original labels for reference
        for i in range(n_windows):
            results[i]['original_labels'] = results[i]['labels'].copy()
        
        # For the first window, keep original labels
        # Then for each subsequent window, match with previous
        for i in range(1, n_windows):
            prev_window = results[i-1]
            curr_window = results[i]
            
            prev_labels = prev_window['labels']
            curr_labels = curr_window['labels']
            
            prev_tickers = prev_window['tickers']
            curr_tickers = curr_window['tickers']
            
            # Find tickers that exist in both windows
            common_tickers = [t for t in prev_tickers if t in curr_tickers]
            
            if not common_tickers:
                continue  # No common tickers, can't track
            
            # Create indices for mapping
            prev_indices = [list(prev_tickers).index(t) for t in common_tickers]
            curr_indices = [list(curr_tickers).index(t) for t in common_tickers]
            
            # Create cluster overlap matrix
            max_prev_cluster = int(np.max(prev_labels))
            max_curr_cluster = int(np.max(curr_labels))
            
            overlap_matrix = np.zeros((max_prev_cluster, max_curr_cluster))
            
            # Calculate overlap between previous and current clusters
            for idx_prev, idx_curr in zip(prev_indices, curr_indices):
                prev_cluster = int(prev_labels[idx_prev]) - 1  # 0-indexed
                curr_cluster = int(curr_labels[idx_curr]) - 1  # 0-indexed
                overlap_matrix[prev_cluster, curr_cluster] += 1
            
            # Use Hungarian algorithm to find optimal matching
            row_ind, col_ind = linear_sum_assignment(-overlap_matrix)
            
            # Create mapping from current labels to new labels
            mapping = {}
            for j, prev_cluster in enumerate(row_ind):
                curr_cluster = col_ind[j]
                # Add 1 to convert back to 1-indexed
                mapping[curr_cluster + 1] = prev_cluster + 1
            
            # Apply mapping to current labels
            new_labels = np.zeros_like(curr_labels)
            for j, label in enumerate(curr_labels):
                if label - 1 in mapping:  # Convert to 0-indexed for lookup
                    new_labels[j] = mapping[label - 1]
                else:
                    # For new clusters, assign a label higher than any previously seen
                    new_labels[j] = max_prev_cluster + (label - len(mapping))
            
            # Update labels
            results[i]['labels'] = new_labels
        
        self.results = results
        return results
    
    def create_cluster_graph(self):
        if self.results is None:
            self.track_clusters()
            
        if not self.results:
            print("No clustering results to create graph from.")
            return nx.DiGraph()
        
        results = self.results
        n_windows = len(results)
        
        # Create a directed graph
        G = nx.DiGraph()
        
        # Add nodes for each cluster in each time window
        for i, result in enumerate(results):
            window_start = result['window'][0]
            labels = result['labels']
            unique_labels = np.unique(labels)
            
            for label in unique_labels:
                # Count stocks in this cluster
                count = np.sum(labels == label)
                # Node ID format: window_index_cluster_label
                node_id = f"w{i}_c{int(label)}"
                G.add_node(node_id, 
                           window=i, 
                           cluster=int(label), 
                           count=int(count),
                           date=window_start)
        
        # Add edges connecting clusters in consecutive windows
        for i in range(n_windows - 1):
            curr_result = results[i]
            next_result = results[i+1]
            
            curr_labels = curr_result['labels']
            next_labels = next_result['labels']
            
            curr_tickers = curr_result['tickers']
            next_tickers = next_result['tickers']
            
            # Find tickers that exist in both windows
            common_tickers = [t for t in curr_tickers if t in next_tickers]
            
            if not common_tickers:
                continue
            
            # Create indices for mapping
            curr_indices = [list(curr_tickers).index(t) for t in common_tickers]
            next_indices = [list(next_tickers).index(t) for t in common_tickers]
            
            # Count transitions between clusters
            transitions = {}
            for curr_idx, next_idx in zip(curr_indices, next_indices):
                curr_cluster = int(curr_labels[curr_idx])
                next_cluster = int(next_labels[next_idx])
                
                key = (f"w{i}_c{curr_cluster}", f"w{i+1}_c{next_cluster}")
                if key in transitions:
                    transitions[key] += 1
                else:
                    transitions[key] = 1
            
            # Add edges
            for (source, target), weight in transitions.items():
                G.add_edge(source, target, weight=weight)
        
        self.cluster_graph = G
        return G
    
    def create_animation(self, output_file='market_clusters_animation.mp4'):
        if self.cluster_graph is None:
            self.create_cluster_graph()
            
        if not self.results:
            print("No results to create animation from.")
            return None
        
        results = self.results
        cluster_graph = self.cluster_graph
        
        # Create figure and axes with larger size to accommodate all stocks in dendrogram
        fig = plt.figure(figsize=(22, 12))
        gs = fig.add_gridspec(2, 3)
        
        ax_tsne = fig.add_subplot(gs[0, 0])      # TSNE plot
        ax_dendro = fig.add_subplot(gs[0, 1:])   # Dendrogram
        ax_flow = fig.add_subplot(gs[1, :])      # Flow diagram
        
        # Create a colormap
        cmap = plt.cm.get_cmap('tab10', 10)  # Up to 10 colors
        
        def update(frame):
            ax_tsne.clear()
            ax_dendro.clear()
            ax_flow.clear()
            
            result = results[frame]
            labels = result['labels']
            tsne_results = result['tsne']
            tickers = result['tickers']
            start_date, end_date = result['window']
            
            # Set title with date range
            fig.suptitle(f"Stock Market Clusters: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}\n"
                        f"Number of clusters: {result['n_clusters']}, Silhouette score: {result['silhouette']:.2f}", 
                        fontsize=16)
            
            # 1. Plot TSNE projection
            unique_labels = np.unique(labels)
            for label in unique_labels:
                cluster_indices = np.where(labels == label)[0]
                label_int = int(label)
                ax_tsne.scatter(tsne_results[cluster_indices, 0], 
                              tsne_results[cluster_indices, 1], 
                              label=f'Cluster {label_int}',
                              color=cmap(label_int % 10),
                              alpha=0.8)
                
                # Add ticker labels for some points
                if len(cluster_indices) < 8:  # Only label if cluster has few members
                    for idx in cluster_indices:
                        ax_tsne.annotate(tickers[idx], (tsne_results[idx, 0], tsne_results[idx, 1]),
                                      fontsize=8, alpha=0.7)
            
            ax_tsne.set_title("t-SNE Projection of Stock Clusters")
            ax_tsne.set_xlabel("t-SNE dimension 1")
            ax_tsne.set_ylabel("t-SNE dimension 2")
            ax_tsne.legend(loc='best')
            
            # 2. Plot dendrogram - show all stocks without threshold
            dendrogram(result['linkage'], ax=ax_dendro, labels=tickers, leaf_rotation=90,
                      color_threshold=None, leaf_font_size=8)
            ax_dendro.set_title("Hierarchical Clustering Dendrogram")
            ax_dendro.set_xlabel("Stocks")
            ax_dendro.set_ylabel("Distance")
            
            # 3. Plot flow diagram showing clusters up to current frame
            # We'll create a simplified flow diagram for the animation
            curr_nodes = [n for n in cluster_graph.nodes() if cluster_graph.nodes[n]['window'] <= frame]
            curr_subgraph = cluster_graph.subgraph(curr_nodes)
            
            pos = {}
            for node in curr_subgraph.nodes():
                window = curr_subgraph.nodes[node]['window']
                cluster = curr_subgraph.nodes[node]['cluster']
                # Position by window (x) and cluster value (y)
                pos[node] = (window, cluster)
            
            # Highlight current frame
            current_nodes = [n for n in curr_nodes if curr_subgraph.nodes[n]['window'] == frame]
            
            # Draw edges with width proportional to transition counts
            for u, v, data in curr_subgraph.edges(data=True):
                ax_flow.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], 
                           'gray', alpha=0.6, linewidth=data['weight']/2)
            
            # Draw nodes
            for node in curr_subgraph.nodes():
                is_current = node in current_nodes
                window = curr_subgraph.nodes[node]['window']
                cluster = curr_subgraph.nodes[node]['cluster']
                count = curr_subgraph.nodes[node]['count']
                
                node_size = count * 20  # Size proportional to number of stocks
                color = cmap(cluster % 10)
                alpha = 1.0 if is_current else 0.5
                
                ax_flow.scatter(window, cluster, s=node_size, color=color, alpha=alpha, 
                              edgecolors='black' if is_current else None)
                
                # Add label showing count for current window
                if is_current:
                    ax_flow.annotate(f"Cluster {int(cluster)}\n{count} stocks", 
                                   (window, cluster), 
                                   xytext=(10, 0), 
                                   textcoords="offset points",
                                   fontsize=9)
            
            # Set axis labels and limits
            ax_flow.set_title("Cluster Evolution Over Time")
            ax_flow.set_xlabel("Time Window")
            ax_flow.set_ylabel("Cluster ID")
            ax_flow.set_xlim(-0.5, len(results) - 0.5)
            ax_flow.set_xticks(range(len(results)))
            ax_flow.set_xticklabels([f"{r['window'][0].strftime('%m/%d/%y')}" for r in results], 
                                  rotation=45, ha='right')
            
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            
            return fig,
        
        ani = animation.FuncAnimation(fig, update, frames=len(results), interval=2000, blit=False)
        
        # Save animation
        try:
            # Try to use ffmpeg writer
            ani.save(output_file, writer='ffmpeg', fps=0.5)
            print(f"Animation saved to '{output_file}'")
        except:
            print("ffmpeg not available. Creating frames and stitching to MP4...")
            try:
                # Create temporary directory for frames
                import tempfile
                import shutil
                import subprocess
                from PIL import Image
                
                temp_dir = tempfile.mkdtemp()
                
                # Save individual frames
                print("Generating frames...")
                for i in range(len(results)):
                    update(i)
                    frame_path = os.path.join(temp_dir, f"frame_{i:03d}.png")
                    plt.savefig(frame_path)
                
                # Try different methods to create MP4
                try:
                    # First try using PIL for GIF (most reliable)
                    gif_output = output_file.replace('.mp4', '.gif')
                    images = []
                    for i in range(len(results)):
                        frame_path = os.path.join(temp_dir, f"frame_{i:03d}.png")
                        images.append(Image.open(frame_path))
                    
                    # Save as GIF with 500ms delay (2fps)
                    images[0].save(
                        gif_output, 
                        save_all=True, 
                        append_images=images[1:], 
                        duration=500, 
                        loop=0
                    )
                    print(f"Animation saved as GIF: '{gif_output}'")
                    
                    # Also try ImageIO for MP4
                    try:
                        import imageio
                        print("Also trying to create MP4 with ImageIO...")
                        
                        frames = []
                        for i in range(len(results)):
                            frame_path = os.path.join(temp_dir, f"frame_{i:03d}.png")
                            frames.append(imageio.imread(frame_path))
                        
                        # Create mp4 with 2 fps (each frame shows for 0.5 seconds)
                        imageio.mimsave(output_file, frames, fps=2)
                        print(f"MP4 animation also saved to '{output_file}' using ImageIO")
                    except Exception as e:
                        print(f"MP4 creation failed: {e}")
                        print("Just use the GIF version instead.")
                
                except Exception as e:
                    print(f"Animation creation failed: {e}")
                
                # Clean up temporary directory
                shutil.rmtree(temp_dir)
                
            except Exception as e:
                print(f"Failed to create animation: {e}")
                plt.close(fig)
        
        return ani
    
    def run_complete_analysis(self, output_file='market_clusters_animation.mp4'):
        # Step 1: Download data
        print("Step 1: Downloading financial data...")
        self.download_data()
        
        if self.data.empty:
            print("Error: No data available for analysis.")
            return
        
        # Step 2: Run clustering
        print("\nStep 2: Running temporal clustering...")
        self.run_clustering()
        
        # Step 3: Track clusters across windows
        print("\nStep 3: Tracking clusters across time windows...")
        self.track_clusters()
        
        # Step 4: Create cluster evolution graph
        print("\nStep 4: Creating cluster evolution graph...")
        self.create_cluster_graph()
        
        # Step 5: Create and save animation
        print("\nStep 5: Creating and saving animation...")
        self.create_animation(output_file)
        
        print("\nAnalysis completed successfully!")

# Example usage
if __name__ == "__main__":
    import argparse
    
    # Command line arguments
    parser = argparse.ArgumentParser(description='Temporal Hierarchical Clustering for Financial Markets')
    parser.add_argument('--tickers', type=int, default=10, help='Number of tickers to analyze')
    parser.add_argument('--start', type=str, default='2022-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default='2022-12-31', help='End date (YYYY-MM-DD)')
    parser.add_argument('--output', type=str, default='market_clusters_animation.mp4', help='Output file name')
    parser.add_argument('--window', type=int, default=60, help='Size of time window in days')
    parser.add_argument('--step', type=int, default=30, help='Step size between windows in days')
    parser.add_argument('--max-clusters', type=int, default=8, help='Maximum number of clusters to consider')
    parser.add_argument('--show-scores', action='store_true', help='Show cluster quality scores for all cluster counts')
    parser.add_argument('--metric', type=str, default='davies_bouldin', choices=['silhouette', 'davies_bouldin', 'calinski_harabasz'], 
                         help='Clustering quality metric to use')
    parser.add_argument('--linkage', type=str, default='average', choices=['ward', 'average', 'complete', 'single'], 
                         help='Linkage method for hierarchical clustering')
    args = parser.parse_args()
    
    # Check if max_clusters is set to a special value
    if args.max_clusters == -1:
        # Set max_clusters to number of tickers (will be constrained internally)
        max_clusters = args.tickers
        print(f"Setting max_clusters to number of tickers: {max_clusters}")
    else:
        max_clusters = args.max_clusters
    
    print(f"Starting analysis with {args.tickers} tickers from {args.start} to {args.end}")
    print(f"Time windows: {args.window} days with {args.step} day steps")
    print(f"Maximum clusters: {max_clusters}")
    print(f"Clustering method: {args.linkage} linkage with {args.metric} metric")
    
    # Create and run the analysis
    analysis = TemporalMarketClustering(
        start_date=args.start,
        end_date=args.end,
        window_size=args.window,
        step_size=args.step,
        max_clusters=max_clusters,
        n_tickers=args.tickers,
        show_all_scores=args.show_scores,
        metric=args.metric,
        linkage_method=args.linkage
    )
    
    # Run the complete analysis pipeline
    analysis.run_complete_analysis(args.output)