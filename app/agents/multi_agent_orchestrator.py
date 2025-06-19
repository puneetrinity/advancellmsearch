"""
Multi-Agent Workflow System
Coordinates specialized AI agents for complex task execution
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import uuid

import structlog
from abc import ABC, abstractmethod

from app.graphs.base import GraphState, NodeResult
from app.models.manager import ModelManager
from app.cache.redis_client import CacheManager
from app.core.config import get_settings

logger = structlog.get_logger(__name__)


class AgentType(Enum):
    """Types of specialized agents"""
    RESEARCH_AGENT = "research"
    ANALYSIS_AGENT = "analysis"
    SYNTHESIS_AGENT = "synthesis"
    FACT_CHECK_AGENT = "fact_check"
    CODE_AGENT = "code"
    CREATIVE_AGENT = "creative"
    PLANNING_AGENT = "planning"
    COORDINATION_AGENT = "coordination"


class AgentStatus(Enum):
    """Agent execution status"""
    IDLE = "idle"
    WORKING = "working"
    COMPLETED = "completed"
    FAILED = "failed"
    WAITING = "waiting"
    BLOCKED = "blocked"


class TaskPriority(Enum):
    """Task priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class AgentTask:
    """Individual task for an agent"""
    task_id: str
    agent_type: AgentType
    task_type: str
    description: str
    input_data: Dict[str, Any]
    dependencies: List[str] = field(default_factory=list)
    priority: TaskPriority = TaskPriority.NORMAL
    timeout: int = 300  # 5 minutes default
    retry_count: int = 0
    max_retries: int = 2
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    status: AgentStatus = AgentStatus.IDLE
    result: Optional[NodeResult] = None

    def is_ready(self, completed_tasks: Set[str]) -> bool:
        return all(dep in completed_tasks for dep in self.dependencies)

    def can_retry(self) -> bool:
        return self.retry_count < self.max_retries

    def update_status(self, status: AgentStatus):
        self.status = status
        self.updated_at = datetime.utcnow()


class BaseAgent(ABC):
    """Abstract base class for specialized agents"""
    def __init__(self, model_manager: ModelManager, cache_manager: CacheManager):
        self.model_manager = model_manager
        self.cache_manager = cache_manager

    @abstractmethod
    async def execute(self, task: AgentTask, state: GraphState) -> NodeResult:
        pass


class ResearchAgent(BaseAgent):
    async def execute(self, task: AgentTask, state: GraphState) -> NodeResult:
        # Placeholder: Implement research logic
        logger.info("ResearchAgent executing", task_id=task.task_id)
        # Simulate research result
        return NodeResult(success=True, data={"research": "Stub research result"}, confidence=0.8)


class AnalysisAgent(BaseAgent):
    async def execute(self, task: AgentTask, state: GraphState) -> NodeResult:
        logger.info("AnalysisAgent executing", task_id=task.task_id)
        return NodeResult(success=True, data={"analysis": "Stub analysis result"}, confidence=0.8)


class SynthesisAgent(BaseAgent):
    async def execute(self, task: AgentTask, state: GraphState) -> NodeResult:
        logger.info("SynthesisAgent executing", task_id=task.task_id)
        return NodeResult(success=True, data={"synthesis": "Stub synthesis result"}, confidence=0.8)


class FactCheckAgent(BaseAgent):
    async def execute(self, task: AgentTask, state: GraphState) -> NodeResult:
        logger.info("FactCheckAgent executing", task_id=task.task_id)
        return NodeResult(success=True, data={"fact_check": "Stub fact check result"}, confidence=0.8)


class CodeAgent(BaseAgent):
    async def execute(self, task: AgentTask, state: GraphState) -> NodeResult:
        logger.info("CodeAgent executing", task_id=task.task_id)
        return NodeResult(success=True, data={"code": "Stub code result"}, confidence=0.8)


class CreativeAgent(BaseAgent):
    async def execute(self, task: AgentTask, state: GraphState) -> NodeResult:
        logger.info("CreativeAgent executing", task_id=task.task_id)
        return NodeResult(success=True, data={"creative": "Stub creative result"}, confidence=0.8)


class PlanningAgent(BaseAgent):
    async def execute(self, task: AgentTask, state: GraphState) -> NodeResult:
        logger.info("PlanningAgent executing", task_id=task.task_id)
        return NodeResult(success=True, data={"planning": "Stub planning result"}, confidence=0.8)


class CoordinationAgent(BaseAgent):
    async def execute(self, task: AgentTask, state: GraphState) -> NodeResult:
        logger.info("CoordinationAgent executing", task_id=task.task_id)
        return NodeResult(success=True, data={"coordination": "Stub coordination result"}, confidence=0.8)


AGENT_CLASS_MAP = {
    AgentType.RESEARCH_AGENT: ResearchAgent,
    AgentType.ANALYSIS_AGENT: AnalysisAgent,
    AgentType.SYNTHESIS_AGENT: SynthesisAgent,
    AgentType.FACT_CHECK_AGENT: FactCheckAgent,
    AgentType.CODE_AGENT: CodeAgent,
    AgentType.CREATIVE_AGENT: CreativeAgent,
    AgentType.PLANNING_AGENT: PlanningAgent,
    AgentType.COORDINATION_AGENT: CoordinationAgent,
}


class MultiAgentOrchestrator:
    """
    Orchestrates multi-agent workflows, resolving dependencies and managing parallel and sequential execution.
    """
    def __init__(self, model_manager: ModelManager, cache_manager: CacheManager):
        self.model_manager = model_manager
        self.cache_manager = cache_manager
        self.settings = get_settings()

    def create_agent(self, agent_type: AgentType) -> BaseAgent:
        agent_cls = AGENT_CLASS_MAP.get(agent_type)
        if not agent_cls:
            raise ValueError(f"No agent class mapped for agent type {agent_type}")
        return agent_cls(self.model_manager, self.cache_manager)

    async def execute_tasks(self, tasks: List[AgentTask], state: Optional[GraphState] = None) -> Dict[str, NodeResult]:
        """
        Executes a list of AgentTasks, resolving dependencies and handling retries.
        Returns a dict mapping task_id to NodeResult.
        """
        pending_tasks = {task.task_id: task for task in tasks}
        completed_tasks: Set[str] = set()
        results: Dict[str, NodeResult] = {}
        state = state or GraphState()

        while pending_tasks:
            # Find all ready tasks
            ready_tasks = [
                task for task in pending_tasks.values()
                if task.is_ready(completed_tasks) and task.status in {AgentStatus.IDLE, AgentStatus.WAITING}
            ]
            if not ready_tasks:
                logger.warning("No ready tasks found, possible circular dependency or all tasks blocked.")
                break

            # Run all ready tasks in parallel
            task_futures = {}
            for task in ready_tasks:
                agent = self.create_agent(task.agent_type)
                task.update_status(AgentStatus.WORKING)
                task_futures[task.task_id] = asyncio.create_task(agent.execute(task, state))

            finished, _ = await asyncio.wait(task_futures.values(), return_when=asyncio.ALL_COMPLETED)

            # Collect results
            for task_id, future in task_futures.items():
                task = pending_tasks[task_id]
                try:
                    result = future.result()
                except Exception as e:
                    logger.error("Agent execution failed", task_id=task_id, error=str(e))
                    task.update_status(AgentStatus.FAILED)
                    if task.can_retry():
                        task.retry_count += 1
                        task.update_status(AgentStatus.WAITING)
                        logger.info("Retrying task", task_id=task_id, retry_count=task.retry_count)
                        continue
                    else:
                        results[task_id] = NodeResult(success=False, error=str(e), confidence=0.0)
                        continue

                if result.success:
                    task.update_status(AgentStatus.COMPLETED)
                    completed_tasks.add(task_id)
                    results[task_id] = result
                else:
                    task.update_status(AgentStatus.FAILED)
                    if task.can_retry():
                        task.retry_count += 1
                        task.update_status(AgentStatus.WAITING)
                        logger.info("Retrying failed task", task_id=task_id, retry_count=task.retry_count)
                        continue
                    else:
                        results[task_id] = result

            # Remove completed tasks from pending
            for task_id in list(completed_tasks):
                if task_id in pending_tasks:
                    del pending_tasks[task_id]

        return results

    def build_task(
        self,
        agent_type: AgentType,
        task_type: str,
        description: str,
        input_data: Dict[str, Any],
        dependencies: Optional[List[str]] = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        timeout: int = 300,
        max_retries: int = 2,
    ) -> AgentTask:
        return AgentTask(
            task_id=str(uuid.uuid4()),
            agent_type=agent_type,
            task_type=task_type,
            description=description,
            input_data=input_data,
            dependencies=dependencies or [],
            priority=priority,
            timeout=timeout,
            max_retries=max_retries
        )
