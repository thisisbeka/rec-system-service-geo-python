"""
Sequential DAGNN-based recommendation algorithm
Predicts next service in a workflow sequence based on DAG structure
"""
import json
import pickle
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from collections import Counter

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import APPNP
from sklearn.preprocessing import LabelEncoder
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.recommendations.base import RecommendationAlgorithm
from app.services.recommendations.models import Recommendation
from app.services.compositions.recovery import recover_new
from app.core.config import settings


class DAGNN(nn.Module):
    """DAGNN module for graph propagation"""
    
    def __init__(self, in_channels: int, K: int, dropout: float = 0.5):
        super().__init__()
        self.propagation = APPNP(K=K, alpha=0.1)
        self.att = nn.Parameter(torch.Tensor(K + 1))
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        self.propagation.reset_parameters()
        nn.init.uniform_(self.att, 0.0, 1.0)

    def forward(self, x, edge_index, training=True):
        xs = [x]
        edge_weight = torch.ones(edge_index.size(1), dtype=torch.float32, device=edge_index.device)
        for _ in range(self.propagation.K):
            x = self.propagation.propagate(edge_index, x=x, edge_weight=edge_weight)
            if training:
                x = F.dropout(x, p=self.dropout, training=training)
            xs.append(x)
        out = torch.stack(xs, dim=-1)
        att_weights = F.softmax(self.att, dim=0)
        out = (out * att_weights.view(1, 1, -1)).sum(dim=-1)
        return out


class DAGNNRecommender(nn.Module):
    """DAGNN-based recommender model"""
    
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, 
                 K: int = 10, dropout: float = 0.4):
        super().__init__()
        self.dropout = dropout
        
        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        
        self.lin2 = nn.Linear(hidden_channels, hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        
        self.dagnn = DAGNN(hidden_channels, K, dropout=dropout)
        
        self.lin3 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.bn3 = nn.BatchNorm1d(hidden_channels // 2)
        
        self.lin_out = nn.Linear(hidden_channels // 2, out_channels)

    def forward(self, x, edge_index, training=True):
        x = self.lin1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=training)
        
        identity = x
        x = self.lin2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = x + identity
        x = F.dropout(x, p=self.dropout, training=training)
        
        x = self.dagnn(x, edge_index, training=training)
        
        x = self.lin3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=training)
        
        x = self.lin_out(x)
        return x


class SequentialDAGNNAlgorithm(RecommendationAlgorithm):
    """
    Sequential recommendation algorithm based on DAGNN
    
    Predicts the next service in a workflow based on:
    - DAG structure from recovered compositions
    - Historical service sequences
    - Graph neural network (DAGNN)
    """
    
    def __init__(
        self,
        db: AsyncSession,
        hidden_channels: int = 64,
        K: int = 10,
        dropout: float = 0.4,
        epochs: int = 200,
        learning_rate: float = 0.001
    ):
        super().__init__(name="sequential_dagnn")
        self.db = db
        self.hidden_channels = hidden_channels
        self.K = K
        self.dropout = dropout
        self.epochs = epochs
        self.learning_rate = learning_rate
        
        # Model components
        self.model: Optional[DAGNNRecommender] = None
        self.dag: Optional[nx.DiGraph] = None
        self.node_map: Optional[Dict] = None
        self.reverse_node_map: Optional[Dict] = None
        self.label_encoder: Optional[LabelEncoder] = None
        self.data_pyg: Optional[Data] = None
        
        # Cache file paths
        self.model_path = Path("app/static/dagnn_model.pth")
        self.metadata_path = Path("app/static/dagnn_metadata.pkl")
    
    async def train(self, data=None) -> None:
        """
        Train the DAGNN model on composition data
        
        Args:
            data: Optional database session (uses self.db if not provided)
        """
        print(f"Training Sequential DAGNN model...")
        
        db = data if data else self.db
        
        # Step 1: Recover compositions using recover_new
        print("Step 1: Recovering compositions...")
        recovery_result = await recover_new(db)
        
        if not recovery_result.get("success"):
            raise ValueError("Failed to recover compositions")
        
        # Step 2: Load DAG from compositions file
        print("Step 2: Loading DAG from compositions...")
        dag_path = Path(settings.CSV_FILE_PATH).parent / "compositionsDAG.json"
        
        if not dag_path.exists():
            raise FileNotFoundError(f"Compositions DAG file not found: {dag_path}")
        
        self.dag = self._load_dag_from_json(dag_path)
        
        # Step 3: Extract paths and create training data
        print("Step 3: Extracting paths from DAG...")
        paths = self._extract_paths_from_dag(self.dag)
        
        if len(paths) == 0:
            raise ValueError("No paths found in DAG")
        
        X_raw, y_raw = self._create_training_data(paths)
        
        if len(X_raw) == 0:
            raise ValueError("No training data created")
        
        # Step 4: Prepare PyTorch Geometric data
        print("Step 4: Preparing PyTorch Geometric data...")
        self.data_pyg, contexts, targets, self.node_map = self._prepare_pytorch_data(
            self.dag, X_raw, y_raw
        )
        
        # Create reverse map (index -> node name)
        self.reverse_node_map = {idx: node for node, idx in self.node_map.items()}
        
        # Step 5: Train DAGNN model
        print(f"Step 5: Training DAGNN model ({self.epochs} epochs)...")
        
        # Initialize model
        self.model = DAGNNRecommender(
            in_channels=2,
            hidden_channels=self.hidden_channels,
            out_channels=len(self.node_map),
            K=self.K,
            dropout=self.dropout
        )
        
        # Train
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=20
        )
        
        best_loss = float('inf')
        patience_counter = 0
        patience = 50
        
        self.model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            
            out = self.model(self.data_pyg.x, self.data_pyg.edge_index, training=True)
            loss = F.cross_entropy(out[contexts], targets)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step(loss)
            
            if loss.item() < best_loss:
                best_loss = loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"   Early stopping at epoch {epoch}")
                    break
            
            if (epoch + 1) % 50 == 0:
                print(f"   Epoch {epoch + 1}/{self.epochs}, Loss: {loss.item():.4f}")
        
        self.is_trained = True
        
        # Save model
        self._save_model()
        
        print(f"✓ Sequential DAGNN trained successfully")
        print(f"   Total nodes: {len(self.node_map)}")
        print(f"   Total paths: {len(paths)}")
        print(f"   Training samples: {len(X_raw)}")
    
    async def recommend(
        self,
        user_id: str,
        n: int = 10,
        exclude_services: Optional[List[int]] = None
    ) -> List[Recommendation]:
        """
        Predict next services in sequence
        
        This method is not applicable for sequential recommendations.
        Use predict_next() instead.
        
        Args:
            user_id: User identifier (not used)
            n: Number of recommendations
            exclude_services: Services to exclude
            
        Returns:
            Empty list (use predict_next for sequential recommendations)
        """
        # Sequential algorithm doesn't work like other recommendation algorithms
        # It needs a sequence as input, not just user_id
        return []
    
    def predict_next(
        self,
        sequence: List[int],
        n: int = 5,
        exclude_services: Optional[List[int]] = None
    ) -> List[Recommendation]:
        """
        Predict next services given a sequence
        
        Args:
            sequence: List of service IDs in the current sequence
            n: Number of recommendations
            exclude_services: Services to exclude from predictions
            
        Returns:
            List of next service recommendations with scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        if len(sequence) == 0:
            # Return most popular starting services
            return self._get_popular_starts(n, exclude_services)
        
        # Convert sequence to node names
        node_sequence = []
        for service_id in sequence:
            node_name = f"service_{service_id}"
            if node_name in self.node_map:
                node_sequence.append(node_name)
        
        if len(node_sequence) == 0:
            return self._get_popular_starts(n, exclude_services)
        
        # Get last node in sequence
        last_node = node_sequence[-1]
        last_node_idx = self.node_map[last_node]
        
        # Predict using model
        self.model.eval()
        with torch.no_grad():
            out = self.model(self.data_pyg.x, self.data_pyg.edge_index, training=False)
            node_scores = out[last_node_idx]
            probs = F.softmax(node_scores, dim=0).numpy()
        
        # Get top predictions
        top_indices = np.argsort(probs)[::-1]
        
        # Filter and create recommendations
        recommendations = []
        exclude_set = set(exclude_services) if exclude_services else set()
        
        # Also exclude services already in sequence
        exclude_set.update(sequence)
        
        for idx in top_indices:
            node_name = self.reverse_node_map[idx]
            
            # Only recommend services (not tables)
            if not node_name.startswith("service_"):
                continue
            
            # Extract service ID
            service_id = int(node_name.split("_")[1])
            
            if service_id in exclude_set:
                continue
            
            # Check if this connection exists in DAG
            if last_node in self.dag and node_name in self.dag.successors(last_node):
                confidence = 0.9  # High confidence - connection exists in DAG
                reason = "dag_connection"
            else:
                confidence = 0.5  # Lower confidence - predicted but not in DAG
                reason = "model_prediction"
            
            recommendations.append(Recommendation(
                service_id=service_id,
                score=float(probs[idx]),
                algorithm=self.name,
                confidence=confidence,
                reason=reason,
                metadata={
                    "sequence_length": len(sequence),
                    "last_service": sequence[-1] if sequence else None,
                    "in_dag": reason == "dag_connection"
                }
            ))
            
            if len(recommendations) >= n:
                break
        
        return recommendations
    
    def _get_popular_starts(self, n: int, exclude_services: Optional[List[int]] = None) -> List[Recommendation]:
        """Get most popular starting services"""
        if not self.dag:
            return []
        
        # Find nodes with no predecessors (starting points)
        start_nodes = [node for node in self.dag.nodes() if self.dag.in_degree(node) == 0]
        
        # Count outgoing edges (popularity)
        node_popularity = {node: self.dag.out_degree(node) for node in start_nodes}
        sorted_nodes = sorted(node_popularity.items(), key=lambda x: x[1], reverse=True)
        
        exclude_set = set(exclude_services) if exclude_services else set()
        recommendations = []
        
        for node, popularity in sorted_nodes:
            if not node.startswith("service_"):
                continue
            
            service_id = int(node.split("_")[1])
            
            if service_id in exclude_set:
                continue
            
            score = popularity / max(node_popularity.values()) if node_popularity.values() else 0.5
            
            recommendations.append(Recommendation(
                service_id=service_id,
                score=score,
                algorithm=self.name,
                confidence=0.7,
                reason="popular_start",
                metadata={"outgoing_edges": popularity}
            ))
            
            if len(recommendations) >= n:
                break
        
        return recommendations
    
    def _load_dag_from_json(self, json_path: Path) -> nx.DiGraph:
        """Load DAG from compositions JSON file"""
        print(f"Loading DAG from {json_path}")
        
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        dag = nx.DiGraph()
        id_to_mid = {}

        for composition in data:
            for node in composition["nodes"]:
                if node.get("mid") is not None:
                    id_to_mid[str(node["id"])] = f"service_{node['mid']}"
                else:
                    id_to_mid[str(node["id"])] = f"table_{node['id']}"

            for link in composition["links"]:
                source = str(link["source"])
                target = str(link["target"])
                src_node = id_to_mid[source]
                tgt_node = id_to_mid[target]
                dag.add_node(src_node, type='service' if src_node.startswith("service") else 'table')
                dag.add_node(tgt_node, type='service' if tgt_node.startswith("service") else 'table')
                dag.add_edge(src_node, tgt_node)

        print(f"Loaded DAG with {dag.number_of_nodes()} nodes and {dag.number_of_edges()} edges")
        return dag
    
    def _extract_paths_from_dag(self, dag: nx.DiGraph) -> List[List[str]]:
        """Extract all paths from DAG"""
        print("Extracting paths from DAG...")
        paths = []

        for start_node in dag.nodes:
            if dag.out_degree(start_node) > 0:
                for path in nx.dfs_edges(dag, source=start_node):
                    full_path = [path[0], path[1]]
                    while dag.out_degree(full_path[-1]) > 0:
                        next_nodes = list(dag.successors(full_path[-1]))
                        if not next_nodes:
                            break
                        full_path.append(next_nodes[0])
                    if len(full_path) > 1:
                        paths.append(full_path)

        print(f"Extracted {len(paths)} paths")
        return paths
    
    def _create_training_data(self, paths: List[List[str]]) -> Tuple[List, List]:
        """Create training samples from paths"""
        X_raw = []
        y_raw = []

        for path in paths:
            for i in range(1, len(path) - 1):
                context = tuple(path[:i])
                next_step = path[i]
                if next_step.startswith("service"):
                    X_raw.append(context)
                    y_raw.append(next_step)

        print(f"Created {len(X_raw)} training samples")
        return X_raw, y_raw
    
    def _prepare_pytorch_data(
        self,
        dag: nx.DiGraph,
        X_raw: List,
        y_raw: List
    ) -> Tuple[Data, torch.Tensor, torch.Tensor, Dict]:
        """Prepare PyTorch Geometric data"""
        print("Preparing PyTorch Geometric data...")
        
        node_list = list(dag.nodes)
        node_encoder = LabelEncoder()
        node_ids = node_encoder.fit_transform(node_list)
        node_map = {node: idx for node, idx in zip(node_list, node_ids)}

        edge_index = torch.tensor(
            [[node_map[u], node_map[v]] for u, v in dag.edges],
            dtype=torch.long
        ).t()
        
        features = [
            [1, 0] if dag.nodes[n]['type'] == 'service' else [0, 1]
            for n in node_list
        ]
        x = torch.tensor(features, dtype=torch.float)
        data_pyg = Data(x=x, edge_index=edge_index)

        contexts = torch.tensor([node_map[context[-1]] for context in X_raw], dtype=torch.long)
        targets = torch.tensor([node_map[y] for y in y_raw], dtype=torch.long)

        return data_pyg, contexts, targets, node_map
    
    def _save_model(self):
        """Save model and metadata to disk"""
        try:
            # Save PyTorch model
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'hidden_channels': self.hidden_channels,
                'K': self.K,
                'dropout': self.dropout,
                'num_nodes': len(self.node_map)
            }, self.model_path)
            
            # Save metadata
            metadata = {
                'node_map': self.node_map,
                'reverse_node_map': self.reverse_node_map,
                'dag_nodes': list(self.dag.nodes()),
                'dag_edges': list(self.dag.edges())
            }
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
            
            print(f"✓ Model saved to {self.model_path}")
            print(f"✓ Metadata saved to {self.metadata_path}")
            
        except Exception as e:
            print(f"Warning: Failed to save model: {e}")
    
    def _load_model(self) -> bool:
        """Load model and metadata from disk"""
        try:
            if not self.model_path.exists() or not self.metadata_path.exists():
                return False
            
            # Load metadata
            with open(self.metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            
            self.node_map = metadata['node_map']
            self.reverse_node_map = metadata['reverse_node_map']
            
            # Reconstruct DAG
            self.dag = nx.DiGraph()
            self.dag.add_nodes_from(metadata['dag_nodes'])
            self.dag.add_edges_from(metadata['dag_edges'])
            
            # Load model
            checkpoint = torch.load(self.model_path)
            self.model = DAGNNRecommender(
                in_channels=2,
                hidden_channels=checkpoint['hidden_channels'],
                out_channels=checkpoint['num_nodes'],
                K=self.K,
                dropout=self.dropout
            )
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            # Recreate data_pyg for inference
            node_list = list(self.dag.nodes())
            edge_index = torch.tensor(
                [[self.node_map[u], self.node_map[v]] for u, v in self.dag.edges()],
                dtype=torch.long
            ).t()
            features = [
                [1, 0] if 'service' in n else [0, 1]
                for n in node_list
            ]
            x = torch.tensor(features, dtype=torch.float)
            self.data_pyg = Data(x=x, edge_index=edge_index)
            
            self.is_trained = True
            print(f"✓ Model loaded from {self.model_path}")
            return True
            
        except Exception as e:
            print(f"Failed to load model: {e}")
            return False
    
    def get_possible_next_services(self, sequence: List[int]) -> List[int]:
        """
        Get possible next services based on DAG structure only
        
        Args:
            sequence: Current sequence of service IDs
            
        Returns:
            List of possible next service IDs from DAG
        """
        if not self.dag or len(sequence) == 0:
            return []
        
        last_service = f"service_{sequence[-1]}"
        
        if last_service not in self.dag:
            return []
        
        # Get successors from DAG
        successors = list(self.dag.successors(last_service))
        
        # Filter only services
        next_services = []
        for node in successors:
            if node.startswith("service_"):
                service_id = int(node.split("_")[1])
                next_services.append(service_id)
        
        return next_services
    
    def predict_next_table(
        self,
        table_sequence: List[int],
        n: int = 5,
        exclude_tables: Optional[List[int]] = None
    ) -> List[Recommendation]:
        """
        Predict next tables (datasets) given a sequence of tables
        
        This method:
        1. Filters only table nodes from the sequence
        2. Ignores service nodes in the DAG path
        3. Predicts next table based on table-to-table patterns in DAG
        
        Args:
            table_sequence: List of table IDs in the current sequence
            n: Number of recommendations
            exclude_tables: Tables to exclude from predictions
            
        Returns:
            List of next table recommendations with scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        if len(table_sequence) == 0:
            # Return most popular starting tables
            return self._get_popular_start_tables(n, exclude_tables)
        
        # Convert table IDs to node names
        table_nodes = []
        for table_id in table_sequence:
            node_name = f"table_{table_id}"
            if node_name in self.node_map:
                table_nodes.append(node_name)
        
        if len(table_nodes) == 0:
            return self._get_popular_start_tables(n, exclude_tables)
        
        # Get last table in sequence
        last_table = table_nodes[-1]
        
        # Find all tables reachable from last_table (through services)
        reachable_tables = self._find_reachable_tables(last_table)
        
        if len(reachable_tables) == 0:
            return []
        
        # Score each reachable table
        table_scores = {}
        
        for table_node in reachable_tables:
            # Calculate score based on:
            # 1. Direct connection distance
            # 2. Frequency in DAG
            # 3. Model prediction
            
            distance = self._get_table_distance(last_table, table_node)
            frequency = self.dag.in_degree(table_node) + self.dag.out_degree(table_node)
            
            # Use model to predict
            last_idx = self.node_map[last_table]
            self.model.eval()
            with torch.no_grad():
                out = self.model(self.data_pyg.x, self.data_pyg.edge_index, training=False)
                probs = F.softmax(out[last_idx], dim=0).numpy()
            
            table_idx = self.node_map[table_node]
            model_score = float(probs[table_idx])
            
            # Combined score
            distance_weight = 1.0 / (distance + 1)  # Closer is better
            frequency_weight = frequency / 10.0  # Normalize
            
            combined_score = (
                0.5 * model_score +
                0.3 * distance_weight +
                0.2 * min(frequency_weight, 1.0)
            )
            
            table_scores[table_node] = {
                'score': combined_score,
                'model_score': model_score,
                'distance': distance,
                'frequency': frequency
            }
        
        # Sort by score
        sorted_tables = sorted(table_scores.items(), key=lambda x: x[1]['score'], reverse=True)
        
        # Build recommendations
        recommendations = []
        exclude_set = set(exclude_tables) if exclude_tables else set()
        exclude_set.update(table_sequence)  # Exclude tables already in sequence
        
        for table_node, scores in sorted_tables:
            # Extract table ID
            table_id = int(table_node.split("_")[1])
            
            if table_id in exclude_set:
                continue
            
            # Determine reason
            if scores['distance'] == 1:
                reason = "direct_connection"
                confidence = 0.9
            elif scores['distance'] <= 3:
                reason = "close_connection"
                confidence = 0.7
            else:
                reason = "distant_connection"
                confidence = 0.5
            
            recommendations.append(Recommendation(
                service_id=table_id,
                score=scores['score'],
                algorithm=self.name,
                confidence=confidence,
                reason=reason,
                metadata={
                    "model_score": round(scores['model_score'], 4),
                    "distance": scores['distance'],
                    "frequency": scores['frequency'],
                    "table_sequence_length": len(table_sequence),
                    "last_table": table_sequence[-1] if table_sequence else None,
                    "type": "table"
                }
            ))
            
            if len(recommendations) >= n:
                break
        
        return recommendations
    
    def _find_reachable_tables(self, start_table: str) -> List[str]:
        """
        Find all tables reachable from start_table through any path in DAG
        
        Args:
            start_table: Starting table node
            
        Returns:
            List of reachable table node names
        """
        if start_table not in self.dag:
            return []
        
        reachable = set()
        visited = set()
        queue = [start_table]
        
        while queue:
            current = queue.pop(0)
            
            if current in visited:
                continue
            
            visited.add(current)
            
            # Get all successors
            for successor in self.dag.successors(current):
                if successor.startswith("table_"):
                    reachable.add(successor)
                
                # Also explore through service nodes
                if successor not in visited:
                    queue.append(successor)
        
        return list(reachable)
    
    def _get_table_distance(self, from_table: str, to_table: str) -> int:
        """
        Get shortest path distance between two tables in DAG
        
        Args:
            from_table: Source table node
            to_table: Target table node
            
        Returns:
            Distance (number of edges), or 999 if not reachable
        """
        try:
            # Only count table nodes in path
            if nx.has_path(self.dag, from_table, to_table):
                all_paths = list(nx.all_simple_paths(self.dag, from_table, to_table, cutoff=10))
                if all_paths:
                    # Count only table nodes in paths
                    min_table_distance = float('inf')
                    for path in all_paths:
                        table_count = sum(1 for node in path if node.startswith("table_"))
                        min_table_distance = min(min_table_distance, table_count - 1)  # -1 because we don't count start
                    return int(min_table_distance)
            return 999
        except:
            return 999
    
    def _get_popular_start_tables(self, n: int, exclude_tables: Optional[List[int]] = None) -> List[Recommendation]:
        """Get most popular starting tables"""
        if not self.dag:
            return []
        
        # Find table nodes with no table predecessors (can have service predecessors)
        table_nodes = [node for node in self.dag.nodes() if node.startswith("table_")]
        
        # Count how often each table appears
        table_frequency = {}
        for table in table_nodes:
            in_degree = self.dag.in_degree(table)
            out_degree = self.dag.out_degree(table)
            table_frequency[table] = in_degree + out_degree
        
        sorted_tables = sorted(table_frequency.items(), key=lambda x: x[1], reverse=True)
        
        exclude_set = set(exclude_tables) if exclude_tables else set()
        recommendations = []
        
        max_freq = sorted_tables[0][1] if sorted_tables else 1
        
        for table_node, frequency in sorted_tables:
            table_id = int(table_node.split("_")[1])
            
            if table_id in exclude_set:
                continue
            
            score = frequency / max_freq if max_freq > 0 else 0.5
            
            recommendations.append(Recommendation(
                service_id=table_id,
                score=score,
                algorithm=self.name,
                confidence=0.6,
                reason="popular_table_start",
                metadata={
                    "frequency": frequency,
                    "type": "table"
                }
            ))
            
            if len(recommendations) >= n:
                break
        
        return recommendations
    
    def get_possible_next_tables(self, table_sequence: List[int]) -> List[int]:
        """
        Get possible next tables based on DAG structure (strict connections)
        
        Args:
            table_sequence: Current sequence of table IDs
            
        Returns:
            List of possible next table IDs from DAG
        """
        if not self.dag or len(table_sequence) == 0:
            return []
        
        last_table = f"table_{table_sequence[-1]}"
        
        if last_table not in self.dag:
            return []
        
        # Find all reachable tables
        reachable = self._find_reachable_tables(last_table)
        
        # Extract IDs
        table_ids = []
        for node in reachable:
            if node.startswith("table_"):
                table_id = int(node.split("_")[1])
                table_ids.append(table_id)
        
        return table_ids
    
    def get_info(self) -> dict:
        """Get algorithm information"""
        info = super().get_info()
        info.update({
            "hidden_channels": self.hidden_channels,
            "K": self.K,
            "dropout": self.dropout,
            "epochs": self.epochs,
            "total_nodes": len(self.node_map) if self.node_map else 0,
            "total_edges": self.dag.number_of_edges() if self.dag else 0,
            "model_saved": self.model_path.exists(),
            "type": "sequential"
        })
        return info

