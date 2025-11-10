"""
Advanced Vibe Matcher - Production-Ready Fashion Recommendation System
Features:
- Multi-modal embeddings (text + image)
- Vector database integration
- Personalization engine
- Advanced filtering
- Real-time learning
- Trend analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import json
import sqlite3
from datetime import datetime, timedelta
import asyncio
import aiohttp
from dataclasses import dataclass, asdict
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class UserProfile:
    """User profile for personalization"""
    user_id: str
    age_group: str = "25-34"
    style_preferences: List[str] = None
    size_preferences: Dict[str, str] = None
    color_preferences: List[str] = None
    price_range: Tuple[float, float] = (0, 1000)
    interaction_history: List[Dict] = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.style_preferences is None:
            self.style_preferences = []
        if self.size_preferences is None:
            self.size_preferences = {}
        if self.color_preferences is None:
            self.color_preferences = []
        if self.interaction_history is None:
            self.interaction_history = []
        if self.created_at is None:
            self.created_at = datetime.now()

@dataclass
class Product:
    """Enhanced product model"""
    id: str
    name: str
    description: str
    category: str
    subcategory: str
    brand: str
    price: float
    colors: List[str]
    sizes: List[str]
    materials: List[str]
    season: str
    gender: str
    vibe_tags: List[str]
    image_url: Optional[str] = None
    text_embedding: Optional[np.ndarray] = None
    image_embedding: Optional[np.ndarray] = None
    popularity_score: float = 0.0
    trend_score: float = 0.0
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

class EmbeddingService:
    """Service for generating text and image embeddings"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.text_model = "text-embedding-ada-002"  # OpenAI model
        self.image_model = "clip-vit-base-patch32"   # CLIP model
        
    async def get_text_embedding(self, text: str) -> np.ndarray:
        """Generate text embedding using OpenAI API"""
        try:
            # Simulate API call - replace with actual OpenAI API call
            # For demo, return random embedding
            return np.random.rand(1536)  # OpenAI embedding dimension
        except Exception as e:
            logger.error(f"Error generating text embedding: {e}")
            return np.random.rand(1536)
    
    async def get_image_embedding(self, image_url: str) -> np.ndarray:
        """Generate image embedding using CLIP model"""
        try:
            # Simulate CLIP embedding - replace with actual CLIP model
            return np.random.rand(512)  # CLIP embedding dimension
        except Exception as e:
            logger.error(f"Error generating image embedding: {e}")
            return np.random.rand(512)
    
    def combine_embeddings(self, text_emb: np.ndarray, image_emb: np.ndarray, 
                          text_weight: float = 0.7) -> np.ndarray:
        """Combine text and image embeddings"""
        # Normalize embeddings
        text_norm = text_emb / np.linalg.norm(text_emb)
        image_norm = image_emb / np.linalg.norm(image_emb)
        
        # Weighted combination
        combined = text_weight * text_norm + (1 - text_weight) * image_norm
        return combined / np.linalg.norm(combined)

class VectorDatabase:
    """Vector database interface for scalable similarity search"""
    
    def __init__(self, db_type: str = "faiss"):
        self.db_type = db_type
        self.index = None
        self.products = {}
        
    def initialize(self, dimension: int):
        """Initialize vector database"""
        if self.db_type == "faiss":
            try:
                import faiss
                self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
                logger.info(f"Initialized FAISS index with dimension {dimension}")
            except ImportError:
                logger.warning("FAISS not available, using numpy fallback")
                self.index = None
        
    def add_vectors(self, vectors: np.ndarray, product_ids: List[str]):
        """Add vectors to database"""
        if self.index is not None:
            # Normalize vectors for cosine similarity
            normalized_vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
            self.index.add(normalized_vectors.astype('float32'))
            
            # Store product mapping
            for i, product_id in enumerate(product_ids):
                self.products[len(self.products)] = product_id
        else:
            # Fallback to numpy storage
            self.vectors = vectors
            self.product_ids = product_ids
    
    def search(self, query_vector: np.ndarray, k: int = 10) -> Tuple[np.ndarray, List[str]]:
        """Search for similar vectors"""
        if self.index is not None:
            # Normalize query vector
            query_norm = query_vector / np.linalg.norm(query_vector)
            scores, indices = self.index.search(query_norm.reshape(1, -1).astype('float32'), k)
            
            product_ids = [self.products[idx] for idx in indices[0]]
            return scores[0], product_ids
        else:
            # Fallback to numpy cosine similarity
            similarities = cosine_similarity(query_vector.reshape(1, -1), self.vectors)[0]
            top_indices = np.argsort(similarities)[::-1][:k]
            
            scores = similarities[top_indices]
            product_ids = [self.product_ids[idx] for idx in top_indices]
            return scores, product_ids

class PersonalizationEngine:
    """Engine for personalized recommendations"""
    
    def __init__(self):
        self.user_profiles = {}
        self.interaction_weights = {
            'view': 1.0,
            'like': 2.0,
            'add_to_cart': 3.0,
            'purchase': 5.0,
            'share': 2.5
        }
        
    def update_user_profile(self, user_id: str, interaction: Dict):
        """Update user profile based on interaction"""
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = UserProfile(user_id=user_id)
        
        profile = self.user_profiles[user_id]
        profile.interaction_history.append({
            **interaction,
            'timestamp': datetime.now()
        })
        
        # Update preferences based on interaction
        self._update_preferences(profile, interaction)
    
    def _update_preferences(self, profile: UserProfile, interaction: Dict):
        """Update user preferences based on interaction"""
        product_data = interaction.get('product_data', {})
        interaction_type = interaction.get('type', 'view')
        weight = self.interaction_weights.get(interaction_type, 1.0)
        
        # Update style preferences
        if 'vibe_tags' in product_data:
            for tag in product_data['vibe_tags']:
                if tag not in profile.style_preferences:
                    profile.style_preferences.append(tag)
        
        # Update color preferences
        if 'colors' in product_data:
            for color in product_data['colors']:
                if color not in profile.color_preferences:
                    profile.color_preferences.append(color)
    
    def get_personalization_vector(self, user_id: str) -> np.ndarray:
        """Generate personalization vector for user"""
        if user_id not in self.user_profiles:
            return np.zeros(100)  # Default neutral vector
        
        profile = self.user_profiles[user_id]
        
        # Create personalization features
        features = []
        
        # Style preference features (50 dimensions)
        style_vector = np.zeros(50)
        for i, pref in enumerate(profile.style_preferences[:50]):
            style_vector[i] = 1.0
        features.extend(style_vector)
        
        # Color preference features (30 dimensions)
        color_vector = np.zeros(30)
        for i, color in enumerate(profile.color_preferences[:30]):
            color_vector[i] = 1.0
        features.extend(color_vector)
        
        # Price preference features (10 dimensions)
        price_min, price_max = profile.price_range
        price_features = [
            price_min / 1000,  # Normalized min price
            price_max / 1000,  # Normalized max price
        ]
        features.extend(price_features + [0] * 8)  # Pad to 10
        
        # Interaction recency features (10 dimensions)
        recent_interactions = [
            int for int in profile.interaction_history 
            if (datetime.now() - int.get('timestamp', datetime.now())).days <= 30
        ]
        recency_features = [len(recent_interactions) / 100] + [0] * 9
        features.extend(recency_features)
        
        return np.array(features[:100])

class TrendAnalyzer:
    """Analyze fashion trends and seasonal patterns"""
    
    def __init__(self):
        self.trend_data = {}
        self.seasonal_patterns = {}
        
    def analyze_trends(self, products: List[Product], interactions: List[Dict]) -> Dict:
        """Analyze current fashion trends"""
        trend_scores = {}
        
        # Analyze interaction patterns
        for interaction in interactions:
            product_id = interaction.get('product_id')
            interaction_type = interaction.get('type', 'view')
            timestamp = interaction.get('timestamp', datetime.now())
            
            # Weight recent interactions more heavily
            days_ago = (datetime.now() - timestamp).days
            recency_weight = max(0.1, 1.0 - (days_ago / 30))
            
            # Calculate trend score
            base_score = {'view': 1, 'like': 2, 'purchase': 5}.get(interaction_type, 1)
            trend_score = base_score * recency_weight
            
            if product_id not in trend_scores:
                trend_scores[product_id] = 0
            trend_scores[product_id] += trend_score
        
        return trend_scores
    
    def get_seasonal_recommendations(self, season: str, user_profile: UserProfile) -> List[str]:
        """Get seasonal product recommendations"""
        seasonal_vibes = {
            'spring': ['fresh', 'light', 'floral', 'pastel', 'breezy'],
            'summer': ['bright', 'airy', 'tropical', 'vibrant', 'casual'],
            'fall': ['cozy', 'warm', 'earthy', 'layered', 'rich'],
            'winter': ['warm', 'luxurious', 'dark', 'heavy', 'festive']
        }
        
        return seasonal_vibes.get(season, [])

class AdvancedVibeMatcherEngine:
    """Advanced Vibe Matcher with all enhanced features"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.embedding_service = EmbeddingService(api_key)
        self.vector_db = VectorDatabase()
        self.personalization_engine = PersonalizationEngine()
        self.trend_analyzer = TrendAnalyzer()
        
        self.products = {}
        self.interaction_history = []
        
        # Initialize vector database
        self.vector_db.initialize(dimension=1536)  # OpenAI embedding dimension
        
    async def add_product(self, product: Product):
        """Add product with embeddings to the system"""
        # Generate embeddings
        product.text_embedding = await self.embedding_service.get_text_embedding(
            f"{product.name} {product.description} {' '.join(product.vibe_tags)}"
        )
        
        if product.image_url:
            product.image_embedding = await self.embedding_service.get_image_embedding(
                product.image_url
            )
            
            # Combine embeddings
            combined_embedding = self.embedding_service.combine_embeddings(
                product.text_embedding, product.image_embedding
            )
            product.text_embedding = combined_embedding
        
        self.products[product.id] = product
        
        # Add to vector database
        self.vector_db.add_vectors(
            product.text_embedding.reshape(1, -1),
            [product.id]
        )
    
    async def search_products(self, 
                            query: str,
                            user_id: Optional[str] = None,
                            filters: Optional[Dict] = None,
                            top_k: int = 10,
                            include_trends: bool = True) -> Dict:
        """Advanced product search with personalization and filtering"""
        
        start_time = datetime.now()
        
        # Generate query embedding
        query_embedding = await self.embedding_service.get_text_embedding(query)
        
        # Apply personalization
        if user_id:
            personalization_vector = self.personalization_engine.get_personalization_vector(user_id)
            # Combine query with personalization (weighted)
            query_embedding = 0.8 * query_embedding + 0.2 * personalization_vector[:len(query_embedding)]
        
        # Search vector database
        scores, product_ids = self.vector_db.search(query_embedding, k=top_k * 2)  # Get more for filtering
        
        # Apply filters
        filtered_results = []
        for score, product_id in zip(scores, product_ids):
            if product_id not in self.products:
                continue
                
            product = self.products[product_id]
            
            # Apply filters
            if filters and not self._passes_filters(product, filters):
                continue
            
            # Calculate final score
            final_score = float(score)
            
            # Add trend boost
            if include_trends:
                trend_scores = self.trend_analyzer.analyze_trends(
                    list(self.products.values()), 
                    self.interaction_history
                )
                trend_boost = trend_scores.get(product_id, 0) * 0.1
                final_score += trend_boost
            
            filtered_results.append({
                'product_id': product_id,
                'name': product.name,
                'description': product.description,
                'brand': product.brand,
                'price': product.price,
                'category': product.category,
                'vibe_tags': product.vibe_tags,
                'colors': product.colors,
                'similarity_score': final_score,
                'trend_score': trend_scores.get(product_id, 0) if include_trends else 0
            })
        
        # Sort by final score and limit results
        filtered_results.sort(key=lambda x: x['similarity_score'], reverse=True)
        filtered_results = filtered_results[:top_k]
        
        search_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return {
            'query': query,
            'user_id': user_id,
            'results': filtered_results,
            'search_time_ms': search_time,
            'total_matches': len(filtered_results),
            'filters_applied': filters or {},
            'personalized': user_id is not None
        }
    
    def _passes_filters(self, product: Product, filters: Dict) -> bool:
        """Check if product passes all filters"""
        
        # Price range filter
        if 'price_range' in filters:
            min_price, max_price = filters['price_range']
            if not (min_price <= product.price <= max_price):
                return False
        
        # Category filter
        if 'categories' in filters:
            if product.category not in filters['categories']:
                return False
        
        # Brand filter
        if 'brands' in filters:
            if product.brand not in filters['brands']:
                return False
        
        # Color filter
        if 'colors' in filters:
            if not any(color in product.colors for color in filters['colors']):
                return False
        
        # Size filter
        if 'sizes' in filters:
            if not any(size in product.sizes for size in filters['sizes']):
                return False
        
        # Season filter
        if 'season' in filters:
            if product.season != filters['season']:
                return False
        
        return True
    
    def record_interaction(self, user_id: str, product_id: str, 
                          interaction_type: str, additional_data: Dict = None):
        """Record user interaction for learning"""
        
        interaction = {
            'user_id': user_id,
            'product_id': product_id,
            'type': interaction_type,
            'timestamp': datetime.now(),
            'product_data': asdict(self.products[product_id]) if product_id in self.products else {},
            **(additional_data or {})
        }
        
        self.interaction_history.append(interaction)
        self.personalization_engine.update_user_profile(user_id, interaction)
    
    def get_trending_products(self, limit: int = 10, time_window_days: int = 7) -> List[Dict]:
        """Get currently trending products"""
        
        cutoff_date = datetime.now() - timedelta(days=time_window_days)
        recent_interactions = [
            int for int in self.interaction_history 
            if int.get('timestamp', datetime.now()) >= cutoff_date
        ]
        
        trend_scores = self.trend_analyzer.analyze_trends(
            list(self.products.values()), 
            recent_interactions
        )
        
        # Sort by trend score
        trending = sorted(trend_scores.items(), key=lambda x: x[1], reverse=True)[:limit]
        
        results = []
        for product_id, score in trending:
            if product_id in self.products:
                product = self.products[product_id]
                results.append({
                    'product_id': product_id,
                    'name': product.name,
                    'brand': product.brand,
                    'price': product.price,
                    'trend_score': score,
                    'vibe_tags': product.vibe_tags
                })
        
        return results
    
    def get_recommendations_for_user(self, user_id: str, limit: int = 10) -> List[Dict]:
        """Get personalized recommendations for user"""
        
        if user_id not in self.personalization_engine.user_profiles:
            return []
        
        profile = self.personalization_engine.user_profiles[user_id]
        
        # Create recommendation query from user preferences
        query_parts = []
        query_parts.extend(profile.style_preferences[:5])  # Top 5 style preferences
        query_parts.extend(profile.color_preferences[:3])  # Top 3 color preferences
        
        query = " ".join(query_parts) if query_parts else "trendy fashion"
        
        # Get recommendations
        results = asyncio.run(self.search_products(
            query=query,
            user_id=user_id,
            filters={
                'price_range': profile.price_range
            },
            top_k=limit,
            include_trends=True
        ))
        
        return results['results']
    
    def export_analytics(self) -> Dict:
        """Export system analytics"""
        
        total_products = len(self.products)
        total_interactions = len(self.interaction_history)
        total_users = len(self.personalization_engine.user_profiles)
        
        # Category distribution
        categories = {}
        for product in self.products.values():
            categories[product.category] = categories.get(product.category, 0) + 1
        
        # Interaction types
        interaction_types = {}
        for interaction in self.interaction_history:
            int_type = interaction.get('type', 'unknown')
            interaction_types[int_type] = interaction_types.get(int_type, 0) + 1
        
        return {
            'total_products': total_products,
            'total_interactions': total_interactions,
            'total_users': total_users,
            'category_distribution': categories,
            'interaction_distribution': interaction_types,
            'avg_interactions_per_user': total_interactions / max(total_users, 1)
        }

# Example usage and testing
async def main():
    """Example usage of the Advanced Vibe Matcher"""
    
    # Initialize the engine
    engine = AdvancedVibeMatcherEngine()
    
    # Add sample products
    sample_products = [
        Product(
            id="1",
            name="Urban Chic Leather Jacket",
            description="Sleek black leather jacket perfect for city adventures",
            category="Outerwear",
            subcategory="Jackets",
            brand="UrbanStyle",
            price=299.99,
            colors=["black", "brown"],
            sizes=["S", "M", "L", "XL"],
            materials=["leather"],
            season="fall",
            gender="unisex",
            vibe_tags=["urban", "edgy", "chic", "rebellious"],
            image_url="https://example.com/jacket.jpg"
        ),
        Product(
            id="2",
            name="Cozy Oversized Sweater",
            description="Soft knit sweater for comfortable lounging",
            category="Tops",
            subcategory="Sweaters",
            brand="ComfortWear",
            price=79.99,
            colors=["cream", "gray", "pink"],
            sizes=["XS", "S", "M", "L"],
            materials=["cotton", "wool"],
            season="winter",
            gender="women",
            vibe_tags=["cozy", "comfortable", "casual", "soft"],
            image_url="https://example.com/sweater.jpg"
        )
    ]
    
    # Add products to engine
    for product in sample_products:
        await engine.add_product(product)
    
    print("🎉 Advanced Vibe Matcher initialized with sample products!")
    
    # Simulate user interactions
    engine.record_interaction("user123", "1", "view")
    engine.record_interaction("user123", "1", "like")
    engine.record_interaction("user123", "2", "view")
    engine.record_interaction("user456", "2", "purchase")
    
    # Test search functionality
    results = await engine.search_products(
        query="edgy urban style",
        user_id="user123",
        filters={"price_range": (0, 500)},
        top_k=5
    )
    
    print("\n🔍 Search Results:")
    for result in results['results']:
        print(f"- {result['name']} (Score: {result['similarity_score']:.3f})")
    
    # Get trending products
    trending = engine.get_trending_products(limit=5)
    print(f"\n📈 Trending Products: {len(trending)} found")
    
    # Get personalized recommendations
    recommendations = engine.get_recommendations_for_user("user123", limit=3)
    print(f"\n👤 Personalized Recommendations: {len(recommendations)} found")
    
    # Export analytics
    analytics = engine.export_analytics()
    print(f"\n📊 System Analytics:")
    print(f"- Total Products: {analytics['total_products']}")
    print(f"- Total Users: {analytics['total_users']}")
    print(f"- Total Interactions: {analytics['total_interactions']}")

if __name__ == "__main__":
    asyncio.run(main())
