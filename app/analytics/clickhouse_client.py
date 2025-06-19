# app/analytics/clickhouse_client.py
"""
ClickHouse Cold Storage Analytics System
Long-term data storage and advanced analytics for AI Search System
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import logging

import asyncio_clickhouse
import structlog
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text

from app.core.config import get_settings
from app.cache.redis_client import CacheManager

logger = structlog.get_logger(__name__)


@dataclass
class QueryExecutionTrace:
    """Complete query execution trace for analytics"""
    query_id: str
    timestamp: datetime
    user_id: str
    user_tier: str
    session_id: str
    query_text: str
    query_type: str
    query_intent: str
    routing_path: List[str]
    models_used: List[str]
    execution_time: float
    total_cost: float
    confidence_scores: Dict[str, float]
    cache_hits: List[str]
    search_providers_used: List[str]
    success: bool
    error_details: Optional[str]
    response_length: int
    quality_requirement: str
    budget_remaining: float
    
    def to_clickhouse_dict(self) -> Dict[str, Any]:
        """Convert to ClickHouse-compatible dictionary"""
        return {
            'query_id': self.query_id,
            'timestamp': self.timestamp,
            'user_id': self.user_id,
            'user_tier': self.user_tier,
            'session_id': self.session_id,
            'query_text': self.query_text,
            'query_type': self.query_type,
            'query_intent': self.query_intent,
            'routing_path': self.routing_path,
            'models_used': self.models_used,
            'execution_time': self.execution_time,
            'total_cost': self.total_cost,
            'confidence_scores': json.dumps(self.confidence_scores),
            'cache_hits': self.cache_hits,
            'search_providers_used': self.search_providers_used,
            'success': self.success,
            'error_details': self.error_details or '',
            'response_length': self.response_length,
            'quality_requirement': self.quality_requirement,
            'budget_remaining': self.budget_remaining
        }


@dataclass
class UserBehaviorPattern:
    """User behavior pattern for analytics"""
    user_id: str
    pattern_type: str
    pattern_data: Dict[str, Any]
    confidence: float
    frequency: int
    last_seen: datetime
    created_at: datetime
    
    def to_clickhouse_dict(self) -> Dict[str, Any]:
        return {
            'user_id': self.user_id,
            'pattern_type': self.pattern_type,
            'pattern_data': json.dumps(self.pattern_data),
            'confidence': self.confidence,
            'frequency': self.frequency,
            'last_seen': self.last_seen,
            'created_at': self.created_at
        }


@dataclass
class SystemMetrics:
    """System performance metrics"""
    timestamp: datetime
    metric_name: str
    metric_value: float
    metric_type: str  # 'performance', 'cost', 'usage', 'quality'
    tags: Dict[str, str]
    
    def to_clickhouse_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'metric_name': self.metric_name,
            'metric_value': self.metric_value,
            'metric_type': self.metric_type,
            'tags': json.dumps(self.tags)
        }


class ClickHouseManager:
    """Manages ClickHouse connection and operations"""
    
    def __init__(self):
        self.settings = get_settings()
        self.client: Optional[asyncio_clickhouse.Client] = None
        self.connection_url = self._build_connection_url()
        self.batch_size = 1000
        self.batch_buffer: List[Dict[str, Any]] = []
        self.last_flush = datetime.now()
        
    def _build_connection_url(self) -> str:
        """Build ClickHouse connection URL"""
        # For development, use local ClickHouse
        # In production, this would be configured via environment variables
        return (
            f"clickhouse://{getattr(self.settings, 'clickhouse_user', 'default')}:"
            f"{getattr(self.settings, 'clickhouse_password', '')}@"
            f"{getattr(self.settings, 'clickhouse_host', 'localhost')}:"
            f"{getattr(self.settings, 'clickhouse_port', 9000)}/"
            f"{getattr(self.settings, 'clickhouse_database', 'ai_search')}"
        )
    
    async def initialize(self):
        """Initialize ClickHouse client and create tables"""
        try:
            self.client = asyncio_clickhouse.Client(
                url=self.connection_url,
                database=getattr(self.settings, 'clickhouse_database', 'ai_search')
            )
            
            # Test connection
            await self.client.execute("SELECT 1")
            
            # Create tables if they don't exist
            await self._create_tables()
            
            # Start background batch processing
            asyncio.create_task(self._batch_processor())
            
            logger.info("ClickHouse manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize ClickHouse: {e}")
            # Fall back to file-based storage for development
            self.client = None
            logger.warning("ClickHouse unavailable - using fallback storage")
    
    async def _create_tables(self):
        """Create ClickHouse tables for analytics"""
        
        # Query execution traces table
        query_traces_sql = """
        CREATE TABLE IF NOT EXISTS query_traces (
            query_id String,
            timestamp DateTime64(3),
            user_id String,
            user_tier LowCardinality(String),
            session_id String,
            query_text String,
            query_type LowCardinality(String),
            query_intent LowCardinality(String),
            routing_path Array(String),
            models_used Array(String),
            execution_time Float64,
            total_cost Float64,
            confidence_scores String,
            cache_hits Array(String),
            search_providers_used Array(String),
            success Bool,
            error_details String,
            response_length UInt32,
            quality_requirement LowCardinality(String),
            budget_remaining Float64
        ) ENGINE = MergeTree()
        ORDER BY (user_id, timestamp)
        PARTITION BY toYYYYMM(timestamp)
        TTL timestamp + INTERVAL 2 YEAR
        """
        
        # Performance metrics table
        performance_metrics_sql = """
        CREATE TABLE IF NOT EXISTS performance_metrics (
            timestamp DateTime64(3),
            metric_name LowCardinality(String),
            metric_value Float64,
            metric_type LowCardinality(String),
            tags String
        ) ENGINE = MergeTree()
        ORDER BY (metric_name, timestamp)
        PARTITION BY toYYYYMM(timestamp)
        TTL timestamp + INTERVAL 1 YEAR
        """
        
        # User behavior patterns table
        user_patterns_sql = """
        CREATE TABLE IF NOT EXISTS user_patterns (
            user_id String,
            pattern_type LowCardinality(String),
            pattern_data String,
            confidence Float64,
            frequency UInt32,
            last_seen DateTime64(3),
            created_at DateTime64(3)
        ) ENGINE = ReplacingMergeTree(last_seen)
        ORDER BY (user_id, pattern_type)
        """
        
        # Model performance table
        model_performance_sql = """
        CREATE TABLE IF NOT EXISTS model_performance (
            timestamp DateTime64(3),
            model_name LowCardinality(String),
            task_type LowCardinality(String),
            execution_time Float64,
            cost Float64,
            confidence Float64,
            success Bool,
            user_tier LowCardinality(String),
            quality_requirement LowCardinality(String)
        ) ENGINE = MergeTree()
        ORDER BY (model_name, timestamp)
        PARTITION BY toYYYYMM(timestamp)
        TTL timestamp + INTERVAL 1 YEAR
        """
        
        # Cache performance table
        cache_performance_sql = """
        CREATE TABLE IF NOT EXISTS cache_performance (
            timestamp DateTime64(3),
            cache_key_hash String,
            cache_strategy LowCardinality(String),
            hit_count UInt32,
            miss_count UInt32,
            avg_response_time Float64,
            cache_size UInt32,
            ttl UInt32
        ) ENGINE = ReplacingMergeTree(timestamp)
        ORDER BY (cache_key_hash, timestamp)
        PARTITION BY toYYYYMM(timestamp)
        TTL timestamp + INTERVAL 6 MONTH
        """
        
        tables = [
            query_traces_sql,
            performance_metrics_sql,
            user_patterns_sql,
            model_performance_sql,
            cache_performance_sql
        ]
        
        for table_sql in tables:
            try:
                await self.client.execute(table_sql)
            except Exception as e:
                logger.error(f"Failed to create table: {e}")
    
    async def insert_query_trace(self, trace: QueryExecutionTrace):
        """Insert query execution trace"""
        if self.client:
            self.batch_buffer.append({
                'table': 'query_traces',
                'data': trace.to_clickhouse_dict()
            })
            await self._check_flush()
        else:
            # Fallback to file storage
            await self._fallback_storage('query_traces', trace.to_clickhouse_dict())
    
    async def insert_performance_metrics(self, metrics: List[SystemMetrics]):
        """Insert performance metrics"""
        if self.client:
            for metric in metrics:
                self.batch_buffer.append({
                    'table': 'performance_metrics',
                    'data': metric.to_clickhouse_dict()
                })
            await self._check_flush()
    
    async def insert_user_pattern(self, pattern: UserBehaviorPattern):
        """Insert user behavior pattern"""
        if self.client:
            self.batch_buffer.append({
                'table': 'user_patterns',
                'data': pattern.to_clickhouse_dict()
            })
            await self._check_flush()
    
    async def _check_flush(self):
        """Check if batch should be flushed"""
        if (len(self.batch_buffer) >= self.batch_size or 
            (datetime.now() - self.last_flush).seconds > 30):
            await self._flush_batch()
    
    async def _flush_batch(self):
        """Flush current batch to ClickHouse"""
        if not self.batch_buffer or not self.client:
            return
        
        try:
            # Group by table
            table_batches = {}
            for item in self.batch_buffer:
                table = item['table']
                if table not in table_batches:
                    table_batches[table] = []
                table_batches[table].append(item['data'])
            
            # Insert each table batch
            for table, data in table_batches.items():
                await self.client.insert(table, data)
            
            logger.debug(f"Flushed {len(self.batch_buffer)} records to ClickHouse")
            self.batch_buffer.clear()
            self.last_flush = datetime.now()
            
        except Exception as e:
            logger.error(f"Failed to flush batch to ClickHouse: {e}")
            # Could implement retry logic here
    
    async def _batch_processor(self):
        """Background batch processor"""
        while True:
            try:
                await asyncio.sleep(30)  # Flush every 30 seconds
                await self._flush_batch()
            except Exception as e:
                logger.error(f"Batch processor error: {e}")
    
    async def _fallback_storage(self, table: str, data: Dict[str, Any]):
        """Fallback storage when ClickHouse is unavailable"""
        # Simple file-based storage for development
        filename = f"data/{table}_{datetime.now().strftime('%Y%m%d')}.jsonl"
        try:
            import os
            os.makedirs("data", exist_ok=True)
            with open(filename, "a") as f:
                f.write(json.dumps(data, default=str) + "\n")
        except Exception as e:
            logger.error(f"Fallback storage failed: {e}")


class AnalyticsEngine:
    """Advanced analytics engine for insights and predictions"""
    
    def __init__(self, clickhouse_manager: ClickHouseManager):
        self.ch = clickhouse_manager
        self.cache_ttl = 3600  # 1 hour cache for analytics
    
    async def get_user_behavior_insights(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive user behavior insights"""
        if not self.ch.client:
            return {"error": "Analytics unavailable"}
        
        try:
            # Query patterns
            query_patterns_sql = """
            SELECT 
                query_type,
                query_intent,
                COUNT(*) as frequency,
                AVG(execution_time) as avg_time,
                AVG(total_cost) as avg_cost,
                AVG(confidence_scores) as avg_confidence
            FROM query_traces 
            WHERE user_id = %s 
                AND timestamp >= now() - INTERVAL 30 DAY
            GROUP BY query_type, query_intent
            ORDER BY frequency DESC
            """
            
            patterns = await self.ch.client.fetch(query_patterns_sql, user_id)
            
            # Usage trends
            usage_trends_sql = """
            SELECT 
                toDate(timestamp) as date,
                COUNT(*) as queries_count,
                SUM(total_cost) as daily_cost,
                AVG(execution_time) as avg_response_time
            FROM query_traces 
            WHERE user_id = %s 
                AND timestamp >= now() - INTERVAL 30 DAY
            GROUP BY date
            ORDER BY date
            """
            
            trends = await self.ch.client.fetch(usage_trends_sql, user_id)
            
            # Model preferences
            model_usage_sql = """
            SELECT 
                arrayJoin(models_used) as model,
                COUNT(*) as usage_count,
                AVG(execution_time) as avg_time,
                SUM(CASE WHEN success THEN 1 ELSE 0 END) / COUNT(*) as success_rate
            FROM query_traces 
            WHERE user_id = %s 
                AND timestamp >= now() - INTERVAL 30 DAY
            GROUP BY model
            ORDER BY usage_count DESC
            """
            
            model_usage = await self.ch.client.fetch(model_usage_sql, user_id)
            
            return {
                "user_id": user_id,
                "analysis_period": "30_days",
                "query_patterns": patterns,
                "usage_trends": trends,
                "model_preferences": model_usage,
                "insights": self._generate_user_insights(patterns, trends, model_usage)
            }
            
        except Exception as e:
            logger.error(f"User behavior analysis failed: {e}")
            return {"error": f"Analysis failed: {str(e)}"}
    
    def _generate_user_insights(self, patterns, trends, model_usage) -> List[str]:
        """Generate actionable insights from user data"""
        insights = []
        
        if patterns:
            top_pattern = patterns[0]
            insights.append(f"Primary query type: {top_pattern['query_type']} ({top_pattern['frequency']} queries)")
            
            if top_pattern['avg_cost'] > 0.02:
                insights.append("Consider using local models more often to reduce costs")
        
        if trends:
            recent_cost = sum(day['daily_cost'] for day in trends[-7:])  # Last 7 days
            if recent_cost > 5.0:
                insights.append("Usage is high - consider upgrading tier for better value")
        
        if model_usage:
            top_model = model_usage[0]
            if top_model['success_rate'] < 0.8:
                insights.append(f"Model {top_model['model']} has low success rate - try different quality settings")
        
        return insights
    
    async def get_system_performance_report(self, time_range: str = "24h") -> Dict[str, Any]:
        """Get comprehensive system performance report"""
        if not self.ch.client:
            return {"error": "Analytics unavailable"}
        
        try:
            # Convert time range
            interval_map = {
                "1h": "1 HOUR",
                "24h": "1 DAY", 
                "7d": "7 DAY",
                "30d": "30 DAY"
            }
            interval = interval_map.get(time_range, "1 DAY")
            
            # Overall metrics
            overview_sql = f"""
            SELECT 
                COUNT(*) as total_queries,
                AVG(execution_time) as avg_response_time,
                percentile(0.95)(execution_time) as p95_response_time,
                SUM(total_cost) as total_cost,
                AVG(total_cost) as avg_cost_per_query,
                SUM(CASE WHEN success THEN 1 ELSE 0 END) / COUNT(*) as success_rate,
                COUNT(DISTINCT user_id) as active_users
            FROM query_traces 
            WHERE timestamp >= now() - INTERVAL {interval}
            """
            
            overview = await self.ch.client.fetchrow(overview_sql)
            
            # Performance by hour
            hourly_perf_sql = f"""
            SELECT 
                toHour(timestamp) as hour,
                COUNT(*) as queries,
                AVG(execution_time) as avg_time,
                percentile(0.95)(execution_time) as p95_time
            FROM query_traces 
            WHERE timestamp >= now() - INTERVAL {interval}
            GROUP BY hour
            ORDER BY hour
            """
            
            hourly_performance = await self.ch.client.fetch(hourly_perf_sql)
            
            # Model performance
            model_perf_sql = f"""
            SELECT 
                arrayJoin(models_used) as model,
                COUNT(*) as usage_count,
                AVG(execution_time) as avg_time,
                SUM(total_cost) as total_cost,
                SUM(CASE WHEN success THEN 1 ELSE 0 END) / COUNT(*) as success_rate
            FROM query_traces 
            WHERE timestamp >= now() - INTERVAL {interval}
            GROUP BY model
            ORDER BY usage_count DESC
            """
            
            model_performance = await self.ch.client.fetch(model_perf_sql)
            
            # Cache performance
            cache_perf_sql = f"""
            SELECT 
                AVG(CASE WHEN length(cache_hits) > 0 THEN 1 ELSE 0 END) as cache_hit_rate,
                AVG(length(cache_hits)) as avg_cache_hits_per_query
            FROM query_traces 
            WHERE timestamp >= now() - INTERVAL {interval}
            """
            
            cache_performance = await self.ch.client.fetchrow(cache_perf_sql)
            
            return {
                "time_range": time_range,
                "overview": dict(overview),
                "hourly_performance": hourly_performance,
                "model_performance": model_performance,
                "cache_performance": dict(cache_performance),
                "recommendations": self._generate_system_recommendations(overview, model_performance)
            }
            
        except Exception as e:
            logger.error(f"System performance analysis failed: {e}")
            return {"error": f"Analysis failed: {str(e)}"}
    
    def _generate_system_recommendations(self, overview, model_performance) -> List[str]:
        """Generate system optimization recommendations"""
        recommendations = []
        
        if overview['avg_response_time'] > 3.0:
            recommendations.append("Average response time is high - consider cache optimization")
        
        if overview['avg_cost_per_query'] > 0.02:
            recommendations.append("Cost per query above target - increase local model usage")
        
        if overview['success_rate'] < 0.95:
            recommendations.append("Success rate below target - review error handling")
        
        # Model-specific recommendations
        if model_performance:
            expensive_models = [m for m in model_performance if m['total_cost'] > 1.0 and m['success_rate'] < 0.8]
            if expensive_models:
                recommendations.append(f"Review usage of expensive low-performing models: {[m['model'] for m in expensive_models]}")
        
        return recommendations
    
    async def predict_user_needs(self, user_id: str) -> Dict[str, Any]:
        """Predict user needs based on historical patterns"""
        if not self.ch.client:
            return {"error": "Prediction unavailable"}
        
        try:
            # Get user's query history
            history_sql = """
            SELECT 
                query_text,
                query_type,
                query_intent,
                timestamp,
                success
            FROM query_traces 
            WHERE user_id = %s 
                AND timestamp >= now() - INTERVAL 7 DAY
            ORDER BY timestamp DESC
            LIMIT 100
            """
            
            history = await self.ch.client.fetch(history_sql, user_id)
            
            if not history:
                return {"predictions": [], "confidence": 0.0}
            
            # Simple pattern analysis (could be enhanced with ML)
            patterns = self._analyze_query_patterns(history)
            predictions = self._generate_predictions(patterns)
            
            return {
                "user_id": user_id,
                "predictions": predictions,
                "confidence": 0.7,  # Would be calculated by ML model
                "based_on_queries": len(history)
            }
            
        except Exception as e:
            logger.error(f"User prediction failed: {e}")
            return {"error": f"Prediction failed: {str(e)}"}
    
    def _analyze_query_patterns(self, history) -> Dict[str, Any]:
        """Analyze patterns in user query history"""
        if not history:
            return {}
        
        # Time patterns
        hours = [datetime.fromisoformat(str(q['timestamp'])).hour for q in history]
        common_hours = max(set(hours), key=hours.count) if hours else 12
        
        # Query type patterns
        types = [q['query_type'] for q in history]
        common_type = max(set(types), key=types.count) if types else 'unknown'
        
        # Intent patterns
        intents = [q['query_intent'] for q in history]
        common_intent = max(set(intents), key=intents.count) if intents else 'unknown'
        
        return {
            "common_hour": common_hours,
            "common_type": common_type,
            "common_intent": common_intent,
            "total_queries": len(history)
        }
    
    def _generate_predictions(self, patterns) -> List[Dict[str, Any]]:
        """Generate predictions based on patterns"""
        predictions = []
        
        if patterns.get("common_type"):
            predictions.append({
                "type": "query_type",
                "prediction": f"Likely to ask {patterns['common_type']} questions",
                "confidence": 0.8,
                "action": "Pre-warm cache for common patterns"
            })
        
        if patterns.get("common_hour"):
            current_hour = datetime.now().hour
            if abs(current_hour - patterns["common_hour"]) <= 1:
                predictions.append({
                    "type": "activity_time",
                    "prediction": "User likely to be active now",
                    "confidence": 0.7,
                    "action": "Optimize response times"
                })
        
        return predictions
    
    async def get_cost_optimization_insights(self) -> Dict[str, Any]:
        """Get cost optimization insights"""
        if not self.ch.client:
            return {"error": "Cost analysis unavailable"}
        
        try:
            # Cost by user tier
            tier_costs_sql = """
            SELECT 
                user_tier,
                COUNT(DISTINCT user_id) as users,
                SUM(total_cost) as total_cost,
                AVG(total_cost) as avg_cost_per_query,
                SUM(total_cost) / COUNT(DISTINCT user_id) as cost_per_user
            FROM query_traces 
            WHERE timestamp >= now() - INTERVAL 30 DAY
            GROUP BY user_tier
            ORDER BY total_cost DESC
            """
            
            tier_costs = await self.ch.client.fetch(tier_costs_sql)
            
            # Most expensive queries
            expensive_queries_sql = """
            SELECT 
                query_type,
                query_intent,
                AVG(total_cost) as avg_cost,
                COUNT(*) as frequency,
                AVG(execution_time) as avg_time
            FROM query_traces 
            WHERE timestamp >= now() - INTERVAL 7 DAY
                AND total_cost > 0
            GROUP BY query_type, query_intent
            HAVING avg_cost > 0.01
            ORDER BY avg_cost DESC
            LIMIT 10
            """
            
            expensive_queries = await self.ch.client.fetch(expensive_queries_sql)
            
            # Local vs API usage
            local_vs_api_sql = """
            SELECT 
                CASE 
                    WHEN total_cost = 0 THEN 'local'
                    ELSE 'api'
                END as processing_type,
                COUNT(*) as queries,
                SUM(total_cost) as total_cost,
                AVG(execution_time) as avg_time
            FROM query_traces 
            WHERE timestamp >= now() - INTERVAL 7 DAY
            GROUP BY processing_type
            """
            
            local_vs_api = await self.ch.client.fetch(local_vs_api_sql)
            
            return {
                "analysis_period": "30_days",
                "tier_analysis": tier_costs,
                "expensive_queries": expensive_queries,
                "local_vs_api": local_vs_api,
                "optimization_opportunities": self._generate_cost_optimization_tips(tier_costs, expensive_queries, local_vs_api)
            }
            
        except Exception as e:
            logger.error(f"Cost analysis failed: {e}")
            return {"error": f"Cost analysis failed: {str(e)}"}
    
    def _generate_cost_optimization_tips(self, tier_costs, expensive_queries, local_vs_api) -> List[str]:
        """Generate cost optimization recommendations"""
        tips = []
        
        # Check local vs API usage
        if local_vs_api:
            total_queries = sum(item['queries'] for item in local_vs_api)
            local_queries = next((item['queries'] for item in local_vs_api if item['processing_type'] == 'local'), 0)
            local_percentage = (local_queries / total_queries * 100) if total_queries > 0 else 0
            
            if local_percentage < 85:
                tips.append(f"Local processing at {local_percentage:.1f}% - target is 85%+")
        
        # Check expensive query patterns
        if expensive_queries:
            top_expensive = expensive_queries[0]
            tips.append(f"Most expensive pattern: {top_expensive['query_type']} queries cost ₹{top_expensive['avg_cost']:.3f} on average")
        
        # Tier optimization
        if tier_costs:
            free_tier_cost = next((item['cost_per_user'] for item in tier_costs if item['user_tier'] == 'free'), 0)
            if free_tier_cost > 15:  # Close to free tier limit
                tips.append("Free tier users approaching cost limits - consider upgrade prompts")
        
        return tips


# Helper function to create analytics system
async def create_analytics_system() -> Tuple[ClickHouseManager, AnalyticsEngine]:
    """Create and initialize analytics system"""
    ch_manager = ClickHouseManager()
    await ch_manager.initialize()
    
    analytics_engine = AnalyticsEngine(ch_manager)
    
    return ch_manager, analytics_engine
