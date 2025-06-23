"""
Advanced startup monitoring and diagnostics for production deployments.
"""
import asyncio
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from app.core.logging import get_logger

logger = get_logger("startup_monitor")


@dataclass
class ComponentInitResult:
    """Result of component initialization."""

    name: str
    status: str  # "success", "failed", "skipped"
    duration: float
    error: Optional[str] = None
    dependencies: List[str] = None
    timestamp: float = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


class StartupMonitor:
    """Advanced startup monitoring with dependency tracking and diagnostics."""

    def __init__(self):
        self.start_time = time.time()
        self.components: Dict[str, ComponentInitResult] = {}
        self.dependency_graph: Dict[str, List[str]] = {
            "model_manager": [],
            "cache_manager": [],
            "chat_graph": ["model_manager", "cache_manager"],
            "search_graph": ["model_manager", "cache_manager"],
            "search_system": ["model_manager", "cache_manager", "search_graph"],
            "api_keys": [],
            # Add more components here as your system grows
            "analytics": [],
            "middleware": [],
            "monitoring": [],
        }

    async def initialize_component(
        self, name: str, init_func, dependencies: List[str] = None
    ):
        """Initialize a component with monitoring and dependency tracking."""
        if dependencies is None:
            dependencies = self.dependency_graph.get(name, [])

        start_time = time.time()
        logger.info(f"🔧 Initializing {name}...")

        # Check dependencies
        missing_deps = []
        for dep in dependencies:
            if dep not in self.components or self.components[dep].status != "success":
                missing_deps.append(dep)

        if missing_deps:
            error_msg = f"Missing dependencies: {missing_deps}"
            self.components[name] = ComponentInitResult(
                name=name,
                status="skipped",
                duration=0.0,
                error=error_msg,
                dependencies=dependencies,
            )
            logger.warning(f"⚠️ Skipping {name}: {error_msg}")
            return None

        # Initialize component
        try:
            if asyncio.iscoroutinefunction(init_func):
                result = await init_func()
            else:
                result = init_func()

            duration = time.time() - start_time
            self.components[name] = ComponentInitResult(
                name=name,
                status="success",
                duration=duration,
                dependencies=dependencies,
            )
            logger.info(f"✅ {name} initialized in {duration:.2f}s")
            return result

        except Exception as e:
            duration = time.time() - start_time
            error_msg = str(e)
            self.components[name] = ComponentInitResult(
                name=name,
                status="failed",
                duration=duration,
                error=error_msg,
                dependencies=dependencies,
            )
            logger.error(f"❌ {name} failed after {duration:.2f}s: {error_msg}")
            return None

    def get_startup_report(self) -> Dict[str, Any]:
        """Generate comprehensive startup report."""
        total_duration = time.time() - self.start_time

        successful = [c for c in self.components.values() if c.status == "success"]
        failed = [c for c in self.components.values() if c.status == "failed"]
        skipped = [c for c in self.components.values() if c.status == "skipped"]

        # Calculate initialization timeline
        timeline = []
        for component in sorted(self.components.values(), key=lambda x: x.timestamp):
            timeline.append(
                {
                    "component": component.name,
                    "status": component.status,
                    "timestamp_offset": component.timestamp - self.start_time,
                    "duration": component.duration,
                    "error": component.error,
                }
            )

        report = {
            "startup_summary": {
                "total_duration": total_duration,
                "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
                "total_components": len(self.components),
                "successful": len(successful),
                "failed": len(failed),
                "skipped": len(skipped),
            },
            "component_details": {
                name: asdict(result) for name, result in self.components.items()
            },
            "initialization_timeline": timeline,
            "dependency_analysis": self._analyze_dependencies(),
            "recommendations": self._generate_recommendations(),
        }
        # Log the full report at debug level for deep troubleshooting
        logger.debug(f"Full startup report: {report}")
        return report

    def _analyze_dependencies(self) -> Dict[str, Any]:
        """Analyze dependency chain and identify issues."""
        analysis = {
            "dependency_chain_healthy": True,
            "broken_chains": [],
            "circular_dependencies": [],
        }

        # Check for broken dependency chains
        for name, result in self.components.items():
            if result.status == "skipped" and result.dependencies:
                failed_deps = [
                    dep
                    for dep in result.dependencies
                    if dep not in self.components
                    or self.components[dep].status != "success"
                ]
                if failed_deps:
                    analysis["broken_chains"].append(
                        {"component": name, "failed_dependencies": failed_deps}
                    )
                    analysis["dependency_chain_healthy"] = False

        return analysis

    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on startup results."""
        recommendations = []

        failed_components = [
            c for c in self.components.values() if c.status == "failed"
        ]
        critical_failures = [
            c for c in failed_components if c.name in ["model_manager"]
        ]

        if critical_failures:
            recommendations.append(
                "🚨 Critical components failed - system functionality severely limited"
            )

        cache_failed = any(
            c.name == "cache_manager" and c.status == "failed"
            for c in failed_components
        )
        if cache_failed:
            recommendations.append(
                "💾 Cache unavailable - expect slower performance and no persistence"
            )

        slow_components = [c for c in self.components.values() if c.duration > 10.0]
        if slow_components:
            recommendations.append(
                f"🐌 Slow initialization detected: {[c.name for c in slow_components]}"
            )

        if not recommendations:
            recommendations.append("🎉 All components initialized successfully!")

        return recommendations
