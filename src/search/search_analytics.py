"""Detailed search analytics and metrics tracking."""

import json
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

from config.logging_config import get_logger

logger = get_logger(__name__)

# Constants
DEFAULT_CACHED_QUERY_LATENCY = 0.01  # 10ms default for cached queries


class SearchMetrics:
    """Container for search metrics data."""

    def __init__(self):
        """Initialize metrics container."""
        self.query_count = 0
        self.total_latency = 0.0
        self.latencies = []
        self.relevance_scores = []
        self.result_counts = []
        self.cache_hits = 0
        self.cache_misses = 0
        self.errors = 0
        self.empty_results = 0

    def add_search(
        self, latency: float, relevance_score: float, result_count: int, from_cache: bool = False
    ):
        """Add a search record to metrics."""
        self.query_count += 1
        self.total_latency += latency
        self.latencies.append(latency)
        self.relevance_scores.append(relevance_score)
        self.result_counts.append(result_count)

        if from_cache:
            self.cache_hits += 1
        else:
            self.cache_misses += 1

        if result_count == 0:
            self.empty_results += 1

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        if not self.query_count:
            return {
                "query_count": 0,
                "avg_latency": 0,
                "avg_relevance": 0,
                "avg_results": 0,
                "cache_hit_rate": 0,
                "empty_result_rate": 0,
            }

        return {
            "query_count": self.query_count,
            "avg_latency": self.total_latency / self.query_count,
            "median_latency": statistics.median(self.latencies) if self.latencies else 0,
            "p95_latency": self._percentile(self.latencies, 95) if self.latencies else 0,
            "avg_relevance": (
                sum(self.relevance_scores) / len(self.relevance_scores)
                if self.relevance_scores
                else 0
            ),
            "avg_results": (
                sum(self.result_counts) / len(self.result_counts) if self.result_counts else 0
            ),
            "cache_hit_rate": (
                self.cache_hits / (self.cache_hits + self.cache_misses)
                if (self.cache_hits + self.cache_misses) > 0
                else 0
            ),
            "empty_result_rate": (
                self.empty_results / self.query_count if self.query_count > 0 else 0
            ),
            "error_rate": self.errors / self.query_count if self.query_count > 0 else 0,
        }

    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile value."""
        if not data:
            return 0
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]


class SearchAnalytics:
    """Service for tracking and analyzing search metrics."""

    def __init__(self, persist_dir: Optional[str] = None, latency_threshold: float = 2.0, relevance_threshold: float = 0.5):
        """
        Initialize search analytics service.

        Args:
            persist_dir: Optional directory for persisting analytics data
            latency_threshold: Threshold for considering queries slow (default: 2.0 seconds)
            relevance_threshold: Minimum relevance score threshold (default: 0.5)
        """
        self.persist_dir = Path(persist_dir) if persist_dir else None

        # Thread-safe storage
        self.lock = Lock()

        # Current session metrics
        self.current_metrics = SearchMetrics()

        # Historical data
        self.search_log = []
        self.query_history = defaultdict(list)
        self.term_frequency = Counter()
        self.failed_queries = []
        self.slow_queries = []

        # Time-based metrics
        self.hourly_metrics = defaultdict(SearchMetrics)
        self.daily_metrics = defaultdict(SearchMetrics)

        # Performance thresholds
        self.latency_threshold = latency_threshold
        self.relevance_threshold = relevance_threshold

        # Load persisted data if available
        if self.persist_dir:
            self._load_persisted_data()

    def track_search(
        self,
        query: str,
        latency: float,
        result_count: int,
        relevance_scores: List[float],
        metadata: Optional[Dict[str, Any]] = None,
        from_cache: bool = False,
        error: Optional[str] = None,
    ) -> None:
        """
        Track a search operation.

        Args:
            query: Search query
            latency: Search latency in seconds
            result_count: Number of results returned
            relevance_scores: List of relevance scores for results
            metadata: Optional metadata about the search
            from_cache: Whether results were from cache
            error: Error message if search failed
        """
        with self.lock:
            timestamp = datetime.utcnow()

            # Calculate average relevance
            avg_relevance = (
                sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0
            )

            # Create log entry
            log_entry = {
                "timestamp": timestamp.isoformat(),
                "query": query,
                "latency": latency,
                "result_count": result_count,
                "avg_relevance": avg_relevance,
                "max_relevance": max(relevance_scores) if relevance_scores else 0.0,
                "from_cache": from_cache,
                "metadata": metadata or {},
                "error": error,
            }

            # Add to search log
            self.search_log.append(log_entry)

            # Update current metrics
            self.current_metrics.add_search(latency, avg_relevance, result_count, from_cache)

            if error:
                self.current_metrics.errors += 1
                self.failed_queries.append(
                    {"query": query, "error": error, "timestamp": timestamp.isoformat()}
                )

            # Update time-based metrics
            hour_key = timestamp.strftime("%Y-%m-%d-%H")
            day_key = timestamp.strftime("%Y-%m-%d")

            self.hourly_metrics[hour_key].add_search(
                latency, avg_relevance, result_count, from_cache
            )
            self.daily_metrics[day_key].add_search(latency, avg_relevance, result_count, from_cache)

            # Track query history
            self.query_history[query].append(
                {
                    "timestamp": timestamp.isoformat(),
                    "latency": latency,
                    "relevance": avg_relevance,
                    "results": result_count,
                }
            )

            # Update term frequency
            terms = query.lower().split()
            self.term_frequency.update(terms)

            # Track slow queries
            if latency > self.latency_threshold:
                self.slow_queries.append(
                    {"query": query, "latency": latency, "timestamp": timestamp.isoformat()}
                )

            # Limit memory usage
            self._cleanup_old_data()

            # Persist if configured
            if self.persist_dir and len(self.search_log) % 100 == 0:
                self._persist_data()

    def get_performance_metrics(self, time_range: Optional[str] = None) -> Dict[str, Any]:
        """
        Get detailed performance metrics.

        Args:
            time_range: Optional time range ('hour', 'day', 'week', 'all')

        Returns:
            Dictionary of performance metrics
        """
        with self.lock:
            if time_range == "hour":
                metrics = self._get_recent_metrics(hours=1)
            elif time_range == "day":
                metrics = self._get_recent_metrics(hours=24)
            elif time_range == "week":
                metrics = self._get_recent_metrics(hours=168)
            else:
                metrics = self.current_metrics

            summary = metrics.get_summary()

            # Add detailed breakdowns
            summary["latency_distribution"] = self._get_latency_distribution(metrics.latencies)
            summary["relevance_distribution"] = self._get_relevance_distribution(
                metrics.relevance_scores
            )
            summary["result_count_distribution"] = self._get_result_distribution(
                metrics.result_counts
            )

            return summary

    def get_query_insights(self, limit: int = 20) -> Dict[str, Any]:
        """
        Get insights about search queries.

        Args:
            limit: Maximum number of items to return in each category

        Returns:
            Dictionary with query insights
        """
        with self.lock:
            # Most popular queries
            popular_queries = Counter()
            for query, history in self.query_history.items():
                popular_queries[query] = len(history)

            # Most popular terms
            popular_terms = self.term_frequency.most_common(limit)

            # Queries with best results
            best_queries = []
            for query, history in self.query_history.items():
                if history:
                    avg_relevance = sum(h["relevance"] for h in history) / len(history)
                    if avg_relevance > self.relevance_threshold:
                        best_queries.append(
                            {
                                "query": query,
                                "avg_relevance": avg_relevance,
                                "search_count": len(history),
                            }
                        )

            best_queries.sort(key=lambda x: x["avg_relevance"], reverse=True)

            return {
                "popular_queries": popular_queries.most_common(limit),
                "popular_terms": popular_terms,
                "best_performing_queries": best_queries[:limit],
                "recent_failed_queries": self.failed_queries[-limit:],
                "recent_slow_queries": self.slow_queries[-limit:],
                "unique_queries": len(self.query_history),
                "total_searches": sum(len(h) for h in self.query_history.values()),
            }

    def get_trend_analysis(
        self, metric: str = "volume", period: str = "hour"
    ) -> List[Dict[str, Any]]:
        """
        Get trend analysis for a specific metric.

        Args:
            metric: Metric to analyze ('volume', 'latency', 'relevance')
            period: Time period ('hour', 'day')

        Returns:
            List of trend data points
        """
        with self.lock:
            if period == "hour":
                metrics_dict = self.hourly_metrics
            else:
                metrics_dict = self.daily_metrics

            trends = []
            for time_key in sorted(metrics_dict.keys()):
                metrics = metrics_dict[time_key]
                summary = metrics.get_summary()

                if metric == "volume":
                    value = summary["query_count"]
                elif metric == "latency":
                    value = summary["avg_latency"]
                elif metric == "relevance":
                    value = summary["avg_relevance"]
                else:
                    value = 0

                trends.append({"time": time_key, "value": value, "metric": metric})

            return trends

    def get_cache_analytics(self) -> Dict[str, Any]:
        """
        Get detailed cache performance analytics.

        Returns:
            Dictionary with cache analytics
        """
        with self.lock:
            total_queries = self.current_metrics.cache_hits + self.current_metrics.cache_misses

            if total_queries == 0:
                return {"cache_hit_rate": 0, "cache_effectiveness": 0, "cached_query_patterns": []}

            # Analyze which queries benefit most from caching
            cached_patterns = []
            for query, history in self.query_history.items():
                cache_hits = sum(1 for h in history if h.get("from_cache", False))
                if cache_hits > 1:
                    cached_patterns.append(
                        {
                            "query": query,
                            "cache_hits": cache_hits,
                            "total_searches": len(history),
                            "cache_rate": cache_hits / len(history),
                        }
                    )

            cached_patterns.sort(key=lambda x: x["cache_hits"], reverse=True)

            # Calculate cache effectiveness (time saved)
            avg_cached_latency = self.current_metrics.average_cached_latency()
            if avg_cached_latency == 0.0:
                avg_cached_latency = DEFAULT_CACHED_QUERY_LATENCY
            avg_uncached_latency = (
                self.current_metrics.total_latency / self.current_metrics.cache_misses
                if self.current_metrics.cache_misses > 0
                else 0
            )
            time_saved = self.current_metrics.cache_hits * (
                avg_uncached_latency - avg_cached_latency
            )

            return {
                "cache_hit_rate": self.current_metrics.cache_hits / total_queries,
                "total_cache_hits": self.current_metrics.cache_hits,
                "total_cache_misses": self.current_metrics.cache_misses,
                "estimated_time_saved": time_saved,
                "cache_effectiveness": (
                    time_saved / self.current_metrics.total_latency
                    if self.current_metrics.total_latency > 0
                    else 0
                ),
                "most_cached_patterns": cached_patterns[:10],
            }

    def get_error_analysis(self) -> Dict[str, Any]:
        """
        Analyze search errors and failures.

        Returns:
            Dictionary with error analysis
        """
        with self.lock:
            if not self.failed_queries:
                return {"error_rate": 0, "common_errors": [], "error_patterns": []}

            # Group errors by type
            error_types = Counter()
            error_queries = defaultdict(list)

            for failure in self.failed_queries:
                error = failure.get("error", "Unknown")
                query = failure.get("query", "")
                error_types[error] += 1
                error_queries[error].append(query)

            # Find patterns in failed queries
            failed_terms = Counter()
            for failure in self.failed_queries:
                query = failure.get("query", "").lower()
                terms = query.split()
                failed_terms.update(terms)

            return {
                "error_rate": (
                    self.current_metrics.errors / self.current_metrics.query_count
                    if self.current_metrics.query_count > 0
                    else 0
                ),
                "total_errors": self.current_metrics.errors,
                "common_errors": error_types.most_common(5),
                "error_patterns": [
                    {"error": error, "example_queries": queries[:3]}
                    for error, queries in list(error_queries.items())[:5]
                ],
                "common_failed_terms": failed_terms.most_common(10),
            }

    def generate_report(self, report_type: str = "summary") -> Dict[str, Any]:
        """
        Generate a comprehensive analytics report.

        Args:
            report_type: Type of report ('summary', 'detailed', 'performance')

        Returns:
            Comprehensive analytics report
        """
        with self.lock:
            base_report = {
                "generated_at": datetime.utcnow().isoformat(),
                "report_type": report_type,
                "period": {
                    "start": self.search_log[0]["timestamp"] if self.search_log else None,
                    "end": self.search_log[-1]["timestamp"] if self.search_log else None,
                },
            }

            if report_type == "summary":
                base_report.update(
                    {
                        "overview": self.current_metrics.get_summary(),
                        "top_queries": self.get_query_insights(10),
                        "cache_performance": self.get_cache_analytics(),
                    }
                )

            elif report_type == "detailed":
                base_report.update(
                    {
                        "overview": self.current_metrics.get_summary(),
                        "performance_metrics": self.get_performance_metrics("all"),
                        "query_insights": self.get_query_insights(20),
                        "cache_analytics": self.get_cache_analytics(),
                        "error_analysis": self.get_error_analysis(),
                        "hourly_trends": self.get_trend_analysis("volume", "hour"),
                        "daily_trends": self.get_trend_analysis("volume", "day"),
                    }
                )

            elif report_type == "performance":
                base_report.update(
                    {
                        "latency_metrics": {
                            "current": self.get_performance_metrics("hour"),
                            "daily": self.get_performance_metrics("day"),
                            "weekly": self.get_performance_metrics("week"),
                        },
                        "latency_trends": self.get_trend_analysis("latency", "hour"),
                        "relevance_trends": self.get_trend_analysis("relevance", "hour"),
                        "slow_queries": self.slow_queries[-20:],
                        "optimization_recommendations": self._get_optimization_recommendations(),
                    }
                )

            return base_report

    def reset_metrics(self, preserve_history: bool = True) -> None:
        """
        Reset current metrics.

        Args:
            preserve_history: Whether to preserve historical data
        """
        with self.lock:
            self.current_metrics = SearchMetrics()

            if not preserve_history:
                self.search_log = []
                self.query_history.clear()
                self.term_frequency.clear()
                self.failed_queries = []
                self.slow_queries = []
                self.hourly_metrics.clear()
                self.daily_metrics.clear()

            logger.info("Search metrics reset")

    def _get_recent_metrics(self, hours: int) -> SearchMetrics:
        """Get metrics for recent time period."""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        recent_metrics = SearchMetrics()

        for entry in self.search_log:
            if datetime.fromisoformat(entry["timestamp"]) > cutoff:
                recent_metrics.add_search(
                    entry["latency"],
                    entry["avg_relevance"],
                    entry["result_count"],
                    entry["from_cache"],
                )

        return recent_metrics

    def _get_latency_distribution(self, latencies: List[float]) -> Dict[str, int]:
        """Get distribution of latency values."""
        if not latencies:
            return {}

        distribution = {"0-100ms": 0, "100-500ms": 0, "500ms-1s": 0, "1s-2s": 0, "2s+": 0}

        for latency in latencies:
            if latency < 0.1:
                distribution["0-100ms"] += 1
            elif latency < 0.5:
                distribution["100-500ms"] += 1
            elif latency < 1.0:
                distribution["500ms-1s"] += 1
            elif latency < 2.0:
                distribution["1s-2s"] += 1
            else:
                distribution["2s+"] += 1

        return distribution

    def _get_relevance_distribution(self, scores: List[float]) -> Dict[str, int]:
        """Get distribution of relevance scores."""
        if not scores:
            return {}

        distribution = {
            "excellent (0.8+)": 0,
            "good (0.6-0.8)": 0,
            "moderate (0.4-0.6)": 0,
            "poor (0.2-0.4)": 0,
            "very poor (<0.2)": 0,
        }

        for score in scores:
            if score >= 0.8:
                distribution["excellent (0.8+)"] += 1
            elif score >= 0.6:
                distribution["good (0.6-0.8)"] += 1
            elif score >= 0.4:
                distribution["moderate (0.4-0.6)"] += 1
            elif score >= 0.2:
                distribution["poor (0.2-0.4)"] += 1
            else:
                distribution["very poor (<0.2)"] += 1

        return distribution

    def _get_result_distribution(self, counts: List[int]) -> Dict[str, int]:
        """Get distribution of result counts."""
        if not counts:
            return {}

        distribution = {
            "0 results": 0,
            "1-5 results": 0,
            "6-10 results": 0,
            "11-20 results": 0,
            "20+ results": 0,
        }

        for count in counts:
            if count == 0:
                distribution["0 results"] += 1
            elif count <= 5:
                distribution["1-5 results"] += 1
            elif count <= 10:
                distribution["6-10 results"] += 1
            elif count <= 20:
                distribution["11-20 results"] += 1
            else:
                distribution["20+ results"] += 1

        return distribution

    def _get_optimization_recommendations(self) -> List[str]:
        """Generate optimization recommendations based on metrics."""
        recommendations = []

        summary = self.current_metrics.get_summary()

        # Check cache performance
        if summary["cache_hit_rate"] < 0.3:
            recommendations.append("Low cache hit rate - consider increasing cache size or TTL")

        # Check latency
        if summary["avg_latency"] > 1.0:
            recommendations.append("High average latency - consider optimizing search indices")

        # Check empty results
        if summary["empty_result_rate"] > 0.2:
            recommendations.append(
                "High empty result rate - consider improving query processing or content coverage"
            )

        # Check error rate
        if summary["error_rate"] > 0.05:
            recommendations.append("High error rate - investigate common error patterns")

        # Check slow queries
        if len(self.slow_queries) > 10:
            recommendations.append(
                f"Found {len(self.slow_queries)} slow queries - consider query optimization"
            )

        return recommendations

    def _cleanup_old_data(self) -> None:
        """Clean up old data to prevent memory issues."""
        # Keep only last 10000 search log entries
        if len(self.search_log) > 10000:
            self.search_log = self.search_log[-10000:]

        # Keep only last 1000 failed/slow queries
        if len(self.failed_queries) > 1000:
            self.failed_queries = self.failed_queries[-1000:]

        if len(self.slow_queries) > 1000:
            self.slow_queries = self.slow_queries[-1000:]

        # Clean up old hourly metrics (keep last 7 days)
        cutoff = datetime.utcnow() - timedelta(days=7)
        cutoff_key = cutoff.strftime("%Y-%m-%d-%H")

        keys_to_remove = [k for k in self.hourly_metrics.keys() if k < cutoff_key]
        for key in keys_to_remove:
            del self.hourly_metrics[key]

        # Clean up old daily metrics (keep last 30 days)
        cutoff = datetime.utcnow() - timedelta(days=30)
        cutoff_key = cutoff.strftime("%Y-%m-%d")

        keys_to_remove = [k for k in self.daily_metrics.keys() if k < cutoff_key]
        for key in keys_to_remove:
            del self.daily_metrics[key]

    def _persist_data(self) -> None:
        """Persist analytics data to disk."""
        if not self.persist_dir:
            return

        try:
            self.persist_dir.mkdir(parents=True, exist_ok=True)

            # Save search log
            log_file = self.persist_dir / f"search_log_{datetime.utcnow().strftime('%Y%m%d')}.json"
            with open(log_file, "w") as f:
                json.dump(self.search_log[-1000:], f)  # Save last 1000 entries

            # Save metrics summary
            summary_file = self.persist_dir / "metrics_summary.json"
            with open(summary_file, "w") as f:
                json.dump(self.generate_report("summary"), f)

            logger.debug(f"Persisted analytics data to {self.persist_dir}")

        except Exception as e:
            logger.error(f"Failed to persist analytics data: {str(e)}")

    def _load_persisted_data(self) -> None:
        """Load persisted analytics data from disk."""
        if not self.persist_dir or not self.persist_dir.exists():
            return

        try:
            # Load most recent search log
            log_files = sorted(self.persist_dir.glob("search_log_*.json"))
            if log_files:
                with open(log_files[-1], "r") as f:
                    loaded_log = json.load(f)
                    self.search_log.extend(loaded_log)

                    # Rebuild metrics from log
                    for entry in loaded_log:
                        self.term_frequency.update(entry["query"].lower().split())

                logger.info(f"Loaded {len(loaded_log)} historical search entries")

        except Exception as e:
            logger.error(f"Failed to load persisted analytics data: {str(e)}")
