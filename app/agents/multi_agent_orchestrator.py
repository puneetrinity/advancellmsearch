"""
Multi-Agent Workflow System
Coordinates specialized AI agents for complex task execution
"""

import asyncio
import json
import re
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import structlog

from app.cache.redis_client import CacheManager
from app.core.config import get_settings
from app.graphs.base import GraphState, NodeResult
from app.models.manager import ModelManager

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
        """Execute research task using AI models and search capabilities"""
        try:
            logger.info(
                "ResearchAgent executing",
                task_id=task.task_id,
                task_type=task.task_type,
                description=task.description,
            )
            # Extract research parameters
            research_query = task.input_data.get("query", state.original_query)
            depth = task.input_data.get("depth", "standard")  # basic, standard, deep
            focus_areas = task.input_data.get("focus_areas", [])
            # Build research prompt based on task type
            if task.task_type == "literature_review":
                prompt = self._build_literature_review_prompt(
                    research_query, focus_areas
                )
            elif task.task_type == "fact_gathering":
                prompt = self._build_fact_gathering_prompt(research_query)
            elif task.task_type == "trend_analysis":
                prompt = self._build_trend_analysis_prompt(research_query)
            else:
                prompt = self._build_general_research_prompt(research_query, depth)
            # Select appropriate model for research
            from app.models.manager import QualityLevel, TaskType

            depth_to_quality = {
                "basic": QualityLevel.MINIMAL,
                "standard": QualityLevel.BALANCED,
                "deep": QualityLevel.PREMIUM,
            }
            quality_level = depth_to_quality.get(depth, QualityLevel.BALANCED)
            model_name = self.model_manager.select_optimal_model(
                TaskType.ANALYTICAL_REASONING, quality_level
            )
            # Generate research findings
            result = await self.model_manager.generate(
                model_name=model_name, prompt=prompt, max_tokens=600, temperature=0.3
            )
            if result.success:
                research_data = self._process_research_findings(
                    result.text, research_query
                )
                if self.cache_manager:
                    cache_key = f"research:{hash(research_query)}"
                    await self.cache_manager.set(cache_key, research_data, ttl=7200)
                logger.info(
                    "Research completed successfully",
                    task_id=task.task_id,
                    findings_count=len(research_data.get("findings", [])),
                )
                return NodeResult(
                    success=True,
                    data={
                        "research_findings": research_data,
                        "methodology": f"{depth} research using {model_name}",
                        "query_analyzed": research_query,
                        "confidence_level": research_data.get("confidence", 0.7),
                    },
                    confidence=0.8,
                    execution_time=result.execution_time,
                    cost=result.cost,
                    model_used=model_name,
                )
            else:
                return NodeResult(
                    success=False,
                    error=f"Research generation failed: {result.error}",
                    confidence=0.0,
                    cost=result.cost,
                )
        except Exception as e:
            logger.error(
                "ResearchAgent execution failed", task_id=task.task_id, error=str(e)
            )
            return NodeResult(
                success=False, error=f"Research agent failed: {str(e)}", confidence=0.0
            )

    def _build_general_research_prompt(self, query: str, depth: str) -> str:
        depth_instructions = {
            "basic": "Provide a concise overview with key facts and 2-3 main points.",
            "standard": "Provide a comprehensive analysis with background, key findings, and implications.",
            "deep": "Provide an in-depth analysis with historical context, multiple perspectives, current trends, and future implications.",
        }
        instruction = depth_instructions.get(depth, depth_instructions["standard"])
        return f"""You are a professional researcher. Conduct thorough research on the following topic:\n\nTopic: {query}\n\nResearch Depth: {depth}\nInstructions: {instruction}\n\nPlease structure your research as follows:\n1. Executive Summary\n2. Key Findings (numbered list)\n3. Supporting Evidence \n4. Implications/Conclusions\n5. Confidence Level (High/Medium/Low)\n\nFocus on accuracy, credibility, and actionable insights. Cite reasoning for your conclusions."""

    def _build_literature_review_prompt(self, query: str, focus_areas: list) -> str:
        focus_text = ", ".join(focus_areas) if focus_areas else "general aspects"
        return f"""Conduct a literature review on: {query}\n\nFocus areas: {focus_text}\n\nProvide:\n1. Overview of current understanding\n2. Key research themes and findings\n3. Gaps in current knowledge\n4. Methodological considerations\n5. Future research directions\n\nStructure your review academically with clear evidence-based conclusions."""

    def _build_fact_gathering_prompt(self, query: str) -> str:
        return f"""Gather and verify key facts about: {query}\n\nProvide:\n1. Verified factual statements (numbered)\n2. Statistical data (if available)\n3. Timeline of key events (if applicable)\n4. Source reliability assessment\n5. Fact confidence ratings\n\nFocus on accuracy and verifiability. Distinguish between facts, claims, and opinions."""

    def _build_trend_analysis_prompt(self, query: str) -> str:
        return f"""Analyze trends related to: {query}\n\nProvide:\n1. Current trend identification\n2. Historical pattern analysis\n3. Driving factors behind trends\n4. Future trend predictions\n5. Potential disruptions or changes\n\nSupport analysis with reasoning and evidence. Indicate confidence levels for predictions."""

    def _process_research_findings(self, research_text: str, query: str) -> dict:
        findings = []
        confidence = 0.7
        sections = research_text.split("\n")
        current_finding = ""
        for line in sections:
            line = line.strip()
            if line and not line.startswith("#"):
                if any(
                    starter in line.lower() for starter in ["1.", "2.", "3.", "-", "•"]
                ):
                    if current_finding:
                        findings.append(current_finding.strip())
                    current_finding = line
                else:
                    current_finding += " " + line
        if current_finding:
            findings.append(current_finding.strip())
        if "high confidence" in research_text.lower():
            confidence = 0.9
        elif "medium confidence" in research_text.lower():
            confidence = 0.7
        elif "low confidence" in research_text.lower():
            confidence = 0.5
        return {
            "query": query,
            "findings": findings[:10],
            "full_text": research_text,
            "confidence": confidence,
            "processed_at": time.time(),
            "finding_count": len(findings),
        }


class AnalysisAgent(BaseAgent):
    async def execute(self, task: AgentTask, state: GraphState) -> NodeResult:
        """Execute analysis task with structured analytical reasoning"""
        try:
            logger.info(
                "AnalysisAgent executing",
                task_id=task.task_id,
                task_type=task.task_type,
            )
            data_to_analyze = task.input_data.get("data", "")
            analysis_type = task.input_data.get("analysis_type", "general")
            context = task.input_data.get("context", state.original_query)
            prompt = self._build_analysis_prompt(
                data_to_analyze, analysis_type, context
            )
            from app.models.manager import QualityLevel, TaskType

            model_name = self.model_manager.select_optimal_model(
                TaskType.ANALYTICAL_REASONING, QualityLevel.BALANCED
            )
            result = await self.model_manager.generate(
                model_name=model_name, prompt=prompt, max_tokens=500, temperature=0.2
            )
            if result.success:
                analysis_data = self._process_analysis_results(
                    result.text, analysis_type
                )
                logger.info(
                    "Analysis completed successfully",
                    task_id=task.task_id,
                    analysis_type=analysis_type,
                )
                return NodeResult(
                    success=True,
                    data={
                        "analysis_results": analysis_data,
                        "analysis_type": analysis_type,
                        "methodology": f"AI-powered {analysis_type} analysis",
                        "key_insights": analysis_data.get("insights", []),
                        "confidence": analysis_data.get("confidence", 0.7),
                    },
                    confidence=0.8,
                    execution_time=result.execution_time,
                    cost=result.cost,
                    model_used=model_name,
                )
            else:
                return NodeResult(
                    success=False,
                    error=f"Analysis generation failed: {result.error}",
                    confidence=0.0,
                    cost=result.cost,
                )
        except Exception as e:
            logger.error(
                "AnalysisAgent execution failed", task_id=task.task_id, error=str(e)
            )
            return NodeResult(
                success=False, error=f"Analysis agent failed: {str(e)}", confidence=0.0
            )

    def _build_analysis_prompt(
        self, data: str, analysis_type: str, context: str
    ) -> str:
        type_prompts = {
            "comparative": f"""Perform a comparative analysis of the following data in the context of: {context}\n\nData to analyze:\n{data}\n\nProvide:\n1. Key similarities and differences\n2. Strengths and weaknesses of each element\n3. Comparative rankings or ratings\n4. Actionable insights\n5. Recommendation based on comparison""",
            "trend": f"""Analyze trends in the following data for: {context}\n\nData to analyze:\n{data}\n\nProvide:\n1. Trend identification (increasing, decreasing, cyclical, etc.)\n2. Rate of change analysis\n3. Contributing factors\n4. Future trend projections\n5. Potential inflection points""",
            "risk": f"""Conduct a risk analysis of the following scenario: {context}\n\nData to analyze:\n{data}\n\nProvide:\n1. Risk identification and categorization\n2. Risk probability assessment (High/Medium/Low)\n3. Impact assessment (High/Medium/Low)\n4. Risk mitigation strategies\n5. Risk monitoring recommendations""",
            "root_cause": f"""Perform root cause analysis for: {context}\n\nData to analyze:\n{data}\n\nProvide:\n1. Problem statement clarification\n2. Potential root causes (ranked by likelihood)\n3. Contributing factors\n4. Evidence supporting each cause\n5. Recommended investigative actions""",
            "swot": f"""Conduct a SWOT analysis for: {context}\n\nData to analyze:\n{data}\n\nProvide:\n1. Strengths (internal positive factors)\n2. Weaknesses (internal negative factors)  \n3. Opportunities (external positive factors)\n4. Threats (external negative factors)\n5. Strategic recommendations based on SWOT""",
            "general": f"""Analyze the following data in the context of: {context}\n\nData to analyze:\n{data}\n\nProvide:\n1. Key patterns and insights\n2. Important relationships or correlations\n3. Anomalies or outliers\n4. Implications and significance\n5. Actionable recommendations""",
        }
        return type_prompts.get(analysis_type, type_prompts["general"])

    def _process_analysis_results(self, analysis_text: str, analysis_type: str) -> dict:
        insights = []
        recommendations = []
        lines = analysis_text.split("\n")
        current_section = None
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if any(
                keyword in line.lower()
                for keyword in ["insight", "finding", "conclusion"]
            ):
                current_section = "insights"
            elif any(
                keyword in line.lower()
                for keyword in ["recommend", "suggest", "action"]
            ):
                current_section = "recommendations"
            if line and (line[0].isdigit() or line.startswith(("-", "•", "*"))):
                if current_section == "insights":
                    insights.append(line)
                elif current_section == "recommendations":
                    recommendations.append(line)
                else:
                    insights.append(line)
        confidence = 0.7
        if len(insights) >= 3 and len(recommendations) >= 2:
            confidence = 0.8
        elif len(insights) >= 5:
            confidence = 0.85
        return {
            "insights": insights[:8],
            "recommendations": recommendations[:5],
            "analysis_type": analysis_type,
            "full_analysis": analysis_text,
            "confidence": confidence,
            "insight_count": len(insights),
            "recommendation_count": len(recommendations),
        }


class SynthesisAgent(BaseAgent):
    async def execute(self, task: AgentTask, state: GraphState) -> NodeResult:
        """Execute synthesis task to combine and integrate multiple sources"""
        try:
            logger.info(
                "SynthesisAgent executing",
                task_id=task.task_id,
                task_type=task.task_type,
            )
            sources = task.input_data.get("sources", [])
            synthesis_goal = task.input_data.get("goal", "comprehensive_overview")
            target_audience = task.input_data.get("audience", "general")
            if hasattr(state, "intermediate_results"):
                research_data = state.intermediate_results.get("research_findings", {})
                analysis_data = state.intermediate_results.get("analysis_results", {})
                if research_data:
                    sources.append({"type": "research", "data": research_data})
                if analysis_data:
                    sources.append({"type": "analysis", "data": analysis_data})
            if not sources:
                return NodeResult(
                    success=False,
                    error="No sources provided for synthesis",
                    confidence=0.0,
                )
            prompt = self._build_synthesis_prompt(
                sources, synthesis_goal, target_audience
            )
            from app.models.manager import QualityLevel, TaskType

            model_name = self.model_manager.select_optimal_model(
                TaskType.ANALYTICAL_REASONING, QualityLevel.BALANCED
            )
            result = await self.model_manager.generate(
                model_name=model_name, prompt=prompt, max_tokens=700, temperature=0.4
            )
            if result.success:
                synthesis_data = self._process_synthesis_results(
                    result.text, synthesis_goal
                )
                logger.info(
                    "Synthesis completed successfully",
                    task_id=task.task_id,
                    sources_count=len(sources),
                )
                return NodeResult(
                    success=True,
                    data={
                        "synthesis_results": synthesis_data,
                        "sources_integrated": len(sources),
                        "synthesis_goal": synthesis_goal,
                        "target_audience": target_audience,
                        "key_themes": synthesis_data.get("themes", []),
                        "confidence": synthesis_data.get("confidence", 0.7),
                    },
                    confidence=0.8,
                    execution_time=result.execution_time,
                    cost=result.cost,
                    model_used=model_name,
                )
            else:
                return NodeResult(
                    success=False,
                    error=f"Synthesis generation failed: {result.error}",
                    confidence=0.0,
                    cost=result.cost,
                )
        except Exception as e:
            logger.error(
                "SynthesisAgent execution failed", task_id=task.task_id, error=str(e)
            )
            return NodeResult(
                success=False, error=f"Synthesis agent failed: {str(e)}", confidence=0.0
            )

    def _build_synthesis_prompt(self, sources: list, goal: str, audience: str) -> str:
        sources_text = ""
        for i, source in enumerate(sources, 1):
            source_type = source.get("type", "unknown")
            source_data = source.get("data", str(source))
            if isinstance(source_data, dict):
                content = source_data.get("content", "")
                if not content:
                    content = str(source_data)[:500]
            else:
                content = str(source_data)[:500]
            sources_text += f"\nSource {i} ({source_type}):\n{content}\n"
        goal_instructions = {
            "comprehensive_overview": "Create a comprehensive overview that integrates all key points from the sources. Identify common themes, resolve contradictions, and present a unified perspective.",
            "executive_summary": "Synthesize the sources into a concise executive summary. Focus on the most important insights, key decisions needed, and actionable recommendations.",
            "comparative_analysis": "Compare and contrast the different perspectives from the sources. Highlight agreements, disagreements, and unique insights from each source.",
            "trend_identification": "Identify and synthesize trends across the sources. Look for patterns, emerging themes, and directional indicators.",
            "decision_support": "Synthesize the information to support decision-making. Weigh different options, assess risks and benefits, and provide clear recommendations.",
        }
        instruction = goal_instructions.get(
            goal, goal_instructions["comprehensive_overview"]
        )
        audience_tone = {
            "executive": "Use executive language, focus on strategic implications and business impact.",
            "technical": "Use precise terminology, include technical details and implementation considerations.",
            "general": "Use clear, accessible language that any educated person can understand.",
            "academic": "Use scholarly tone with rigorous analysis and evidence-based conclusions.",
        }
        tone = audience_tone.get(audience, audience_tone["general"])
        return f"""You are a synthesis expert. Your task is to integrate multiple sources into a cohesive, valuable synthesis.\n\nGoal: {goal}\nTarget Audience: {audience}\nTone: {tone}\n\nSources to synthesize:\n{sources_text}\n\nInstructions: {instruction}\n\nPlease provide your synthesis with:\n1. Executive Summary\n2. Key Integrated Insights\n3. Common Themes\n4. Conflicting Viewpoints (if any)\n5. Synthesis Conclusions\n6. Confidence Level in synthesis\n\nEnsure the synthesis adds value beyond just summarizing individual sources."""

    def _process_synthesis_results(self, synthesis_text: str, goal: str) -> dict:
        themes = []
        insights = []
        conclusions = []
        lines = synthesis_text.split("\n")
        current_section = None
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if any(
                keyword in line.lower() for keyword in ["theme", "pattern", "trend"]
            ):
                current_section = "themes"
            elif any(
                keyword in line.lower()
                for keyword in ["insight", "finding", "observation"]
            ):
                current_section = "insights"
            elif any(
                keyword in line.lower()
                for keyword in ["conclusion", "summary", "outcome"]
            ):
                current_section = "conclusions"
            if line and (line[0].isdigit() or line.startswith(("-", "•", "*"))):
                if current_section == "themes":
                    themes.append(line)
                elif current_section == "insights":
                    insights.append(line)
                elif current_section == "conclusions":
                    conclusions.append(line)
        confidence = 0.7
        if len(themes) >= 2 and len(insights) >= 3:
            confidence = 0.8
        if len(conclusions) >= 2:
            confidence = min(confidence + 0.1, 0.9)
        return {
            "themes": themes[:5],
            "insights": insights[:8],
            "conclusions": conclusions[:5],
            "full_synthesis": synthesis_text,
            "goal": goal,
            "confidence": confidence,
            "synthesis_quality": "high" if confidence > 0.8 else "medium",
        }


class FactCheckAgent(BaseAgent):
    async def execute(self, task: AgentTask, state: GraphState) -> NodeResult:
        """Execute fact-checking task to verify claims and statements"""
        try:
            logger.info(
                "FactCheckAgent executing",
                task_id=task.task_id,
                task_type=task.task_type,
            )
            claims_to_check = task.input_data.get("claims", [])
            text_to_analyze = task.input_data.get("text", "")
            verification_level = task.input_data.get("level", "standard")
            if not claims_to_check and text_to_analyze:
                claims_to_check = self._extract_claims_from_text(text_to_analyze)
            if not claims_to_check:
                return NodeResult(
                    success=False,
                    error="No claims provided for fact-checking",
                    confidence=0.0,
                )
            fact_check_results = []
            total_cost = 0.0
            for claim in claims_to_check[:5]:
                result = await self._verify_single_claim(claim, verification_level)
                fact_check_results.append(result)
                total_cost += result.get("cost", 0.0)
            verified_count = sum(
                1 for r in fact_check_results if r["status"] == "verified"
            )
            disputed_count = sum(
                1 for r in fact_check_results if r["status"] == "disputed"
            )
            uncertain_count = sum(
                1 for r in fact_check_results if r["status"] == "uncertain"
            )
            overall_confidence = (
                verified_count / len(fact_check_results) if fact_check_results else 0.0
            )
            logger.info(
                "Fact-checking completed",
                task_id=task.task_id,
                claims_checked=len(fact_check_results),
                verified=verified_count,
                disputed=disputed_count,
            )
            return NodeResult(
                success=True,
                data={
                    "fact_check_results": fact_check_results,
                    "summary": {
                        "total_claims": len(fact_check_results),
                        "verified": verified_count,
                        "disputed": disputed_count,
                        "uncertain": uncertain_count,
                        "verification_rate": verified_count / len(fact_check_results)
                        if fact_check_results
                        else 0.0,
                    },
                    "verification_level": verification_level,
                    "overall_reliability": self._calculate_reliability_score(
                        fact_check_results
                    ),
                },
                confidence=overall_confidence,
                cost=total_cost,
            )
        except Exception as e:
            logger.error(
                "FactCheckAgent execution failed", task_id=task.task_id, error=str(e)
            )
            return NodeResult(
                success=False,
                error=f"Fact-check agent failed: {str(e)}",
                confidence=0.0,
            )

    async def _verify_single_claim(self, claim: str, verification_level: str) -> dict:
        try:
            prompt = self._build_verification_prompt(claim, verification_level)
            from app.models.manager import QualityLevel, TaskType

            level_to_quality = {
                "basic": QualityLevel.MINIMAL,
                "standard": QualityLevel.BALANCED,
                "thorough": QualityLevel.PREMIUM,
            }
            quality_level = level_to_quality.get(
                verification_level, QualityLevel.BALANCED
            )
            model_name = self.model_manager.select_optimal_model(
                TaskType.ANALYTICAL_REASONING, quality_level
            )
            result = await self.model_manager.generate(
                model_name=model_name, prompt=prompt, max_tokens=300, temperature=0.1
            )
            if result.success:
                verification_data = self._parse_verification_result(result.text)
                return {
                    "claim": claim,
                    "status": verification_data["status"],
                    "confidence": verification_data["confidence"],
                    "reasoning": verification_data["reasoning"],
                    "evidence": verification_data.get("evidence", []),
                    "cost": result.cost,
                    "model_used": model_name,
                }
            else:
                return {
                    "claim": claim,
                    "status": "error",
                    "confidence": 0.0,
                    "reasoning": f"Verification failed: {result.error}",
                    "evidence": [],
                    "cost": result.cost,
                }
        except Exception as e:
            return {
                "claim": claim,
                "status": "error",
                "confidence": 0.0,
                "reasoning": f"Verification error: {str(e)}",
                "evidence": [],
                "cost": 0.0,
            }

    def _extract_claims_from_text(self, text: str) -> list:
        sentences = text.split(".")
        claims = []
        fact_indicators = [
            "is",
            "are",
            "was",
            "were",
            "has",
            "have",
            "will",
            "can",
            "does",
            "did",
            "shows",
            "proves",
            "demonstrates",
            "indicates",
            "according to",
            "studies show",
            "research indicates",
            "data shows",
        ]
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20 and any(
                indicator in sentence.lower() for indicator in fact_indicators
            ):
                claims.append(sentence)
        return claims[:5]

    def _build_verification_prompt(self, claim: str, level: str) -> str:
        level_instructions = {
            "basic": "Provide a quick assessment of whether this claim is likely true, false, or uncertain.",
            "standard": "Analyze this claim for factual accuracy. Consider common knowledge, logical consistency, and plausibility.",
            "thorough": "Conduct a detailed analysis of this claim. Consider multiple perspectives, potential sources of error, and confidence levels.",
        }
        instruction = level_instructions.get(level, level_instructions["standard"])
        return f"""You are a fact-checker. Analyze the following claim for accuracy.\n\nClaim to verify: \"{claim}\"\n\nInstructions: {instruction}\n\nPlease provide your assessment in this format:\nSTATUS: [VERIFIED/DISPUTED/UNCERTAIN]\nCONFIDENCE: [0.0-1.0]\nREASONING: [Your detailed reasoning]\nEVIDENCE: [Any supporting or contradicting evidence you can identify]\n\nBe objective and indicate your confidence level. If uncertain, explain why."""

    def _parse_verification_result(self, verification_text: str) -> dict:
        lines = verification_text.strip().split("\n")
        result = {
            "status": "uncertain",
            "confidence": 0.5,
            "reasoning": verification_text,
            "evidence": [],
        }
        for line in lines:
            line = line.strip()
            if line.startswith("STATUS:"):
                status_text = line.replace("STATUS:", "").strip().lower()
                if "verified" in status_text or "true" in status_text:
                    result["status"] = "verified"
                elif "disputed" in status_text or "false" in status_text:
                    result["status"] = "disputed"
                else:
                    result["status"] = "uncertain"
            elif line.startswith("CONFIDENCE:"):
                try:
                    conf_text = line.replace("CONFIDENCE:", "").strip()
                    confidence = float(conf_text)
                    result["confidence"] = max(0.0, min(1.0, confidence))
                except ValueError:
                    pass
            elif line.startswith("REASONING:"):
                result["reasoning"] = line.replace("REASONING:", "").strip()
            elif line.startswith("EVIDENCE:"):
                evidence_text = line.replace("EVIDENCE:", "").strip()
                if evidence_text:
                    result["evidence"] = [evidence_text]
        return result

    def _calculate_reliability_score(self, fact_check_results: list) -> str:
        if not fact_check_results:
            return "unknown"
        verified_count = sum(1 for r in fact_check_results if r["status"] == "verified")
        disputed_count = sum(1 for r in fact_check_results if r["status"] == "disputed")
        total_count = len(fact_check_results)
        verified_rate = verified_count / total_count
        disputed_rate = disputed_count / total_count
        if verified_rate >= 0.8:
            return "high"
        elif verified_rate >= 0.6 and disputed_rate <= 0.2:
            return "medium"
        elif disputed_rate >= 0.4:
            return "low"
        else:
            return "mixed"


class CodeAgent(BaseAgent):
    async def execute(self, task: AgentTask, state: GraphState) -> NodeResult:
        """Execute code-related tasks including generation, review, and debugging"""
        try:
            logger.info(
                "CodeAgent executing", task_id=task.task_id, task_type=task.task_type
            )
            code_request = task.input_data.get("request", "")
            programming_language = task.input_data.get("language", "python")
            code_to_review = task.input_data.get("existing_code", "")
            difficulty_level = task.input_data.get("difficulty", "intermediate")
            if task.task_type == "code_generation":
                prompt = self._build_code_generation_prompt(
                    code_request, programming_language, difficulty_level
                )
            elif task.task_type == "code_review":
                prompt = self._build_code_review_prompt(
                    code_to_review, programming_language
                )
            elif task.task_type == "debugging":
                prompt = self._build_debugging_prompt(
                    code_to_review, code_request, programming_language
                )
            elif task.task_type == "optimization":
                prompt = self._build_optimization_prompt(
                    code_to_review, programming_language
                )
            else:
                prompt = self._build_general_code_prompt(
                    code_request, programming_language
                )
            from app.models.manager import QualityLevel, TaskType

            model_name = self.model_manager.select_optimal_model(
                TaskType.CODE_TASKS, QualityLevel.BALANCED
            )
            result = await self.model_manager.generate(
                model_name=model_name, prompt=prompt, max_tokens=600, temperature=0.2
            )
            if result.success:
                code_data = self._process_code_result(
                    result.text, task.task_type, programming_language
                )
                logger.info(
                    "Code task completed successfully",
                    task_id=task.task_id,
                    task_type=task.task_type,
                    language=programming_language,
                )
                return NodeResult(
                    success=True,
                    data={
                        "code_results": code_data,
                        "task_type": task.task_type,
                        "programming_language": programming_language,
                        "code_quality": code_data.get("quality_score", "good"),
                        "has_executable_code": code_data.get("has_code", False),
                        "explanation": code_data.get("explanation", ""),
                    },
                    confidence=0.8,
                    execution_time=result.execution_time,
                    cost=result.cost,
                    model_used=model_name,
                )
            else:
                return NodeResult(
                    success=False,
                    error=f"Code generation failed: {result.error}",
                    confidence=0.0,
                    cost=result.cost,
                )
        except Exception as e:
            logger.error(
                "CodeAgent execution failed", task_id=task.task_id, error=str(e)
            )
            return NodeResult(
                success=False, error=f"Code agent failed: {str(e)}", confidence=0.0
            )

    def _build_code_generation_prompt(
        self, request: str, language: str, difficulty: str
    ) -> str:
        difficulty_guidance = {
            "beginner": "Use simple, well-commented code with clear variable names. Explain each step.",
            "intermediate": "Use appropriate design patterns and best practices. Include error handling.",
            "advanced": "Optimize for performance and maintainability. Use advanced language features appropriately.",
        }
        guidance = difficulty_guidance.get(
            difficulty, difficulty_guidance["intermediate"]
        )
        return f"""You are an expert {language} programmer. Generate clean, working code for the following request.\n\nRequest: {request}\nProgramming Language: {language}\nDifficulty Level: {difficulty}\nGuidance: {guidance}\n\nPlease provide:\n1. Complete, working code\n2. Clear comments explaining the logic\n3. Usage example\n4. Any important notes or considerations\n\nMake sure the code is production-ready and follows {language} best practices."""

    def _build_code_review_prompt(self, code: str, language: str) -> str:
        return f"""You are a senior code reviewer. Review the following {language} code for quality, security, and best practices.\n\nCode to review:\n```{language}\n{code}\n```\n\nPlease provide:\n1. Overall code quality assessment\n2. Specific issues found (bugs, security vulnerabilities, style issues)\n3. Suggestions for improvement\n4. Performance considerations\n5. Maintainability assessment\n6. Rating: (Excellent/Good/Fair/Poor)\n\nBe constructive and specific in your feedback."""

    def _build_debugging_prompt(
        self, code: str, issue_description: str, language: str
    ) -> str:
        return f"""You are a debugging expert. Help identify and fix issues in the following {language} code.\n\nIssue Description: {issue_description}\n\nCode with issues:\n```{language}\n{code}\n```\n\nPlease provide:\n1. Identification of the bug(s)\n2. Explanation of why the bug occurs\n3. Fixed version of the code\n4. Prevention strategies for similar issues\n5. Testing recommendations\n\nFocus on providing a working solution with clear explanations."""

    def _build_optimization_prompt(self, code: str, language: str) -> str:
        return f"""You are a performance optimization expert. Optimize the following {language} code for better performance, memory usage, and efficiency.\n\nCode to optimize:\n```{language}\n{code}\n```\n\nPlease provide:\n1. Performance analysis of current code\n2. Optimized version of the code\n3. Explanation of optimizations made\n4. Performance improvements expected\n5. Trade-offs (if any)\n\nFocus on meaningful improvements while maintaining readability."""

    def _build_general_code_prompt(self, request: str, language: str) -> str:
        return f"""You are a helpful {language} programming assistant. Help with the following coding request:\n\nRequest: {request}\n\nPlease provide:\n1. Relevant code solution or guidance\n2. Clear explanation of the approach\n3. Example usage (if applicable)\n4. Best practices advice\n5. Common pitfalls to avoid\n\nMake your response practical and actionable."""

    def _process_code_result(
        self, code_text: str, task_type: str, language: str
    ) -> dict:
        code_blocks = re.findall(r"```(?:\w+)?\n(.*?)\n```", code_text, re.DOTALL)
        explanation_parts = re.split(r"```(?:\w+)?\n.*?\n```", code_text)
        explanation = " ".join(
            part.strip() for part in explanation_parts if part.strip()
        )
        quality_indicators = {
            "has_comments": bool(re.search(r"#.*|//.*|/\*.*\*/", code_text)),
            "has_error_handling": bool(
                re.search(r"try|except|catch|error|Error", code_text, re.IGNORECASE)
            ),
            "has_functions": bool(re.search(r"def |function |class ", code_text)),
            "has_documentation": bool(
                re.search(r'""".*?"""|\'\'\'.*?\'\'\'', code_text, re.DOTALL)
            ),
        }
        quality_score = sum(quality_indicators.values()) / len(quality_indicators)
        if quality_score >= 0.75:
            quality_rating = "excellent"
        elif quality_score >= 0.5:
            quality_rating = "good"
        elif quality_score >= 0.25:
            quality_rating = "fair"
        else:
            quality_rating = "basic"
        return {
            "code_blocks": code_blocks,
            "explanation": explanation,
            "has_code": len(code_blocks) > 0,
            "quality_score": quality_rating,
            "quality_indicators": quality_indicators,
            "task_type": task_type,
            "language": language,
            "line_count": sum(len(block.split("\n")) for block in code_blocks),
            "complexity": "high"
            if len(code_blocks) > 1
            else "medium"
            if code_blocks
            else "low",
        }


class CreativeAgent(BaseAgent):
    async def execute(self, task: AgentTask, state: GraphState) -> NodeResult:
        """Execute creative tasks including writing, brainstorming, and content generation"""
        try:
            logger.info(
                "CreativeAgent executing",
                task_id=task.task_id,
                task_type=task.task_type,
            )
            creative_request = task.input_data.get("request", "")
            content_type = task.input_data.get("content_type", "general")
            tone = task.input_data.get("tone", "professional")
            target_audience = task.input_data.get("audience", "general")
            length = task.input_data.get("length", "medium")
            if task.task_type == "writing":
                prompt = self._build_writing_prompt(
                    creative_request, content_type, tone, target_audience, length
                )
            elif task.task_type == "brainstorming":
                prompt = self._build_brainstorming_prompt(
                    creative_request, content_type
                )
            elif task.task_type == "storytelling":
                prompt = self._build_storytelling_prompt(
                    creative_request, tone, target_audience
                )
            elif task.task_type == "marketing_copy":
                prompt = self._build_marketing_prompt(
                    creative_request, tone, target_audience
                )
            else:
                prompt = self._build_general_creative_prompt(
                    creative_request, content_type, tone
                )
            from app.models.manager import QualityLevel, TaskType

            model_name = self.model_manager.select_optimal_model(
                TaskType.CREATIVE_WRITING, QualityLevel.BALANCED
            )
            result = await self.model_manager.generate(
                model_name=model_name, prompt=prompt, max_tokens=500, temperature=0.7
            )
            if result.success:
                creative_data = self._process_creative_result(
                    result.text, task.task_type, content_type
                )
                logger.info(
                    "Creative task completed successfully",
                    task_id=task.task_id,
                    task_type=task.task_type,
                    content_type=content_type,
                )
                return NodeResult(
                    success=True,
                    data={
                        "creative_results": creative_data,
                        "task_type": task.task_type,
                        "content_type": content_type,
                        "tone": tone,
                        "target_audience": target_audience,
                        "creativity_score": creative_data.get("creativity_score", 0.7),
                        "word_count": creative_data.get("word_count", 0),
                    },
                    confidence=0.8,
                    execution_time=result.execution_time,
                    cost=result.cost,
                    model_used=model_name,
                )
            else:
                return NodeResult(
                    success=False,
                    error=f"Creative generation failed: {result.error}",
                    confidence=0.0,
                    cost=result.cost,
                )
        except Exception as e:
            logger.error(
                "CreativeAgent execution failed", task_id=task.task_id, error=str(e)
            )
            return NodeResult(
                success=False, error=f"Creative agent failed: {str(e)}", confidence=0.0
            )

    def _build_writing_prompt(
        self, request: str, content_type: str, tone: str, audience: str, length: str
    ) -> str:
        length_guidance = {
            "short": "Keep it concise, around 100-200 words",
            "medium": "Provide a substantial piece, around 300-500 words",
            "long": "Create a comprehensive piece, 500+ words",
        }
        length_guide = length_guidance.get(length, length_guidance["medium"])
        return f"""You are a skilled creative writer. Create engaging content based on the following specifications.\n\nRequest: {request}\nContent Type: {content_type}\nTone: {tone}\nTarget Audience: {audience}\nLength: {length} ({length_guide})\n\nPlease create content that:\n1. Captures attention from the beginning\n2. Maintains the specified tone throughout\n3. Engages the target audience effectively\n4. Has a clear structure and flow\n5. Ends with impact\n\nMake it creative, original, and compelling."""

    def _build_brainstorming_prompt(self, request: str, content_type: str) -> str:
        return f"""You are a creative brainstorming facilitator. Generate innovative ideas for the following challenge.\n\nChallenge: {request}\nContent Type: {content_type}\n\nPlease provide:\n1. 5-7 creative and diverse ideas\n2. Brief explanation for each idea\n3. Potential benefits and unique aspects\n4. Implementation considerations\n5. One "wild card" innovative idea\n\nThink outside the box and be creative. No idea is too bold for brainstorming."""

    def _build_storytelling_prompt(self, request: str, tone: str, audience: str) -> str:
        return f"""You are a master storyteller. Create an engaging story based on the following.\n\nStory Request: {request}\nTone: {tone}\nAudience: {audience}\n\nPlease craft a story that:\n1. Has a compelling beginning that hooks the reader\n2. Develops interesting characters (if applicable)\n3. Maintains good pacing and flow\n4. Includes vivid descriptions and details\n5. Has a satisfying conclusion\n\nMake it memorable and emotionally engaging."""

    def _build_marketing_prompt(self, request: str, tone: str, audience: str) -> str:
        return f"""You are a creative marketing copywriter. Create compelling marketing content.\n\nMarketing Brief: {request}\nTone: {tone}\nTarget Audience: {audience}\n\nPlease create marketing copy that:\n1. Grabs attention immediately\n2. Clearly communicates value proposition\n3. Addresses audience pain points or desires\n4. Includes persuasive elements\n5. Has a strong call-to-action\n\nMake it persuasive, authentic, and results-oriented."""

    def _build_general_creative_prompt(
        self, request: str, content_type: str, tone: str
    ) -> str:
        return f"""You are a versatile creative professional. Help with this creative challenge.\n\nCreative Request: {request}\nContent Type: {content_type}\nTone: {tone}\n\nPlease provide creative content that:\n1. Addresses the request thoroughly\n2. Demonstrates originality and creativity\n3. Maintains appropriate tone\n4. Is well-structured and engaging\n5. Adds unique value or perspective\n\nBe innovative and think creatively."""

    def _process_creative_result(
        self, creative_text: str, task_type: str, content_type: str
    ) -> dict:
        word_count = len(creative_text.split())
        sentence_count = len([s for s in creative_text.split(".") if s.strip()])
        paragraph_count = len([p for p in creative_text.split("\n\n") if p.strip()])
        creativity_indicators = {
            "uses_metaphors": bool(
                re.search(r"like|as |metaphor|symboliz", creative_text, re.IGNORECASE)
            ),
            "has_vivid_imagery": bool(
                re.search(
                    r"color|bright|dark|visual|image|picture",
                    creative_text,
                    re.IGNORECASE,
                )
            ),
            "shows_emotion": bool(
                re.search(
                    r"feel|emotion|heart|soul|passion|excit",
                    creative_text,
                    re.IGNORECASE,
                )
            ),
            "has_varied_sentences": sentence_count > 5
            and word_count / sentence_count > 10,
            "uses_active_voice": creative_text.count("ing ")
            > creative_text.count(" was ") + creative_text.count(" were "),
        }
        creativity_score = sum(creativity_indicators.values()) / len(
            creativity_indicators
        )
        themes = []
        for sentence in creative_text.split("."):
            sentence = sentence.strip()
            if len(sentence) > 30:
                themes.append(sentence[:50] + "..." if len(sentence) > 50 else sentence)
        return {
            "content": creative_text,
            "word_count": word_count,
            "sentence_count": sentence_count,
            "paragraph_count": paragraph_count,
            "creativity_score": creativity_score,
            "creativity_indicators": creativity_indicators,
            "task_type": task_type,
            "content_type": content_type,
            "themes": themes[:3],
            "readability": "high" if word_count / sentence_count < 20 else "medium",
            "engagement_level": "high" if creativity_score > 0.6 else "medium",
        }


class PlanningAgent(BaseAgent):
    async def execute(self, task: AgentTask, state: GraphState) -> NodeResult:
        """Execute planning tasks including project planning, strategy development, and task organization"""
        try:
            logger.info(
                "PlanningAgent executing",
                task_id=task.task_id,
                task_type=task.task_type,
            )
            planning_objective = task.input_data.get("objective", "")
            time_horizon = task.input_data.get("time_horizon", "medium")
            complexity_level = task.input_data.get("complexity", "medium")
            constraints = task.input_data.get("constraints", [])
            resources = task.input_data.get("resources", [])
            if task.task_type == "project_planning":
                prompt = self._build_project_planning_prompt(
                    planning_objective, time_horizon, constraints, resources
                )
            elif task.task_type == "strategic_planning":
                prompt = self._build_strategic_planning_prompt(
                    planning_objective, time_horizon, constraints
                )
            elif task.task_type == "task_breakdown":
                prompt = self._build_task_breakdown_prompt(
                    planning_objective, complexity_level
                )
            elif task.task_type == "resource_planning":
                prompt = self._build_resource_planning_prompt(
                    planning_objective, resources, constraints
                )
            else:
                prompt = self._build_general_planning_prompt(
                    planning_objective, time_horizon, complexity_level
                )
            from app.models.manager import QualityLevel, TaskType

            model_name = self.model_manager.select_optimal_model(
                TaskType.ANALYTICAL_REASONING, QualityLevel.BALANCED
            )
            result = await self.model_manager.generate(
                model_name=model_name, prompt=prompt, max_tokens=600, temperature=0.3
            )
            if result.success:
                planning_data = self._process_planning_result(
                    result.text, task.task_type, time_horizon
                )
                logger.info(
                    "Planning task completed successfully",
                    task_id=task.task_id,
                    task_type=task.task_type,
                    time_horizon=time_horizon,
                )
                return NodeResult(
                    success=True,
                    data={
                        "planning_results": planning_data,
                        "task_type": task.task_type,
                        "time_horizon": time_horizon,
                        "complexity_level": complexity_level,
                        "total_tasks": planning_data.get("task_count", 0),
                        "estimated_timeline": planning_data.get("timeline", ""),
                        "feasibility_score": planning_data.get("feasibility", 0.7),
                    },
                    confidence=0.8,
                    execution_time=result.execution_time,
                    cost=result.cost,
                    model_used=model_name,
                )
            else:
                return NodeResult(
                    success=False,
                    error=f"Planning generation failed: {result.error}",
                    confidence=0.0,
                    cost=result.cost,
                )
        except Exception as e:
            logger.error(
                "PlanningAgent execution failed", task_id=task.task_id, error=str(e)
            )
            return NodeResult(
                success=False, error=f"Planning agent failed: {str(e)}", confidence=0.0
            )

    def _build_project_planning_prompt(
        self, objective: str, time_horizon: str, constraints: list, resources: list
    ) -> str:
        constraints_text = ", ".join(constraints) if constraints else "None specified"
        resources_text = ", ".join(resources) if resources else "Standard resources"
        time_guidance = {
            "short": "Focus on immediate actions and quick wins (days to weeks)",
            "medium": "Balance quick wins with substantial progress (weeks to months)",
            "long": "Include strategic phases and long-term milestones (months to years)",
        }
        guidance = time_guidance.get(time_horizon, time_guidance["medium"])
        return f"""You are a professional project manager. Create a comprehensive project plan.\n\nProject Objective: {objective}\nTime Horizon: {time_horizon} ({guidance})\nConstraints: {constraints_text}\nAvailable Resources: {resources_text}\n\nPlease provide a structured project plan with:\n1. Project scope and deliverables\n2. Work breakdown structure (major phases and tasks)\n3. Timeline estimates for each phase\n4. Resource allocation requirements\n5. Risk assessment and mitigation strategies\n6. Success criteria and milestones\n7. Dependencies and critical path items\n\nMake it actionable and realistic."""

    def _build_strategic_planning_prompt(
        self, objective: str, time_horizon: str, constraints: list
    ) -> str:
        constraints_text = ", ".join(constraints) if constraints else "None specified"
        return f"""You are a strategic planning consultant. Develop a strategic plan for achieving this objective.\n\nStrategic Objective: {objective}\nTime Horizon: {time_horizon}\nKey Constraints: {constraints_text}\n\nPlease provide a strategic plan including:\n1. Strategic analysis (current state, desired state)\n2. Key strategic initiatives (3-5 major areas)\n3. Strategic priorities and sequencing\n4. Resource and capability requirements\n5. Risk assessment and contingency planning\n6. Success metrics and KPIs\n7. Implementation roadmap\n\nFocus on high-level strategy with actionable direction."""

    def _build_task_breakdown_prompt(
        self, objective: str, complexity_level: str
    ) -> str:
        complexity_guidance = {
            "simple": "Break into 3-5 clear, manageable tasks",
            "medium": "Create a detailed breakdown with 5-10 tasks and subtasks",
            "complex": "Provide comprehensive breakdown with multiple levels and dependencies",
        }
        guidance = complexity_guidance.get(
            complexity_level, complexity_guidance["medium"]
        )
        return f"""You are a task management expert. Break down this objective into actionable tasks.\n\nObjective: {objective}\nComplexity Level: {complexity_level}\nGuidance: {guidance}\n\nPlease provide:\n1. Main task categories\n2. Specific actionable tasks for each category\n3. Task priorities (High/Medium/Low)\n4. Estimated effort for each task\n5. Task dependencies and sequence\n6. Deliverables for each task\n\nMake each task specific, measurable, and actionable."""

    def _build_resource_planning_prompt(
        self, objective: str, resources: list, constraints: list
    ) -> str:
        resources_text = ", ".join(resources) if resources else "Standard resources"
        constraints_text = ", ".join(constraints) if constraints else "None specified"
        return f"""You are a resource planning specialist. Plan resource allocation for this objective.\n\nObjective: {objective}\nAvailable Resources: {resources_text}\nConstraints: {constraints_text}\n\nPlease provide resource planning including:\n1. Resource requirements analysis\n2. Resource allocation plan\n3. Resource optimization strategies\n4. Alternative resource options\n5. Resource scheduling and timing\n6. Budget considerations (if applicable)\n7. Resource risk mitigation\n\nFocus on efficient and realistic resource utilization."""

    def _build_general_planning_prompt(
        self, objective: str, time_horizon: str, complexity_level: str
    ) -> str:
        return f"""You are a planning expert. Create a comprehensive plan for achieving this objective.\n\nObjective: {objective}\nTime Horizon: {time_horizon}\nComplexity Level: {complexity_level}\n\nPlease provide:\n1. Goal clarification and scope definition\n2. Key phases and milestones\n3. Detailed action steps\n4. Timeline and scheduling\n5. Resource and support needs\n6. Potential challenges and solutions\n7. Success measurements\n\nMake it practical, achievable, and well-structured."""

    def _process_planning_result(
        self, planning_text: str, task_type: str, time_horizon: str
    ) -> dict:
        tasks = []
        phases = []
        milestones = []
        lines = planning_text.split("\n")
        current_section = None
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if any(keyword in line.lower() for keyword in ["phase", "stage", "step"]):
                current_section = "phases"
            elif any(
                keyword in line.lower() for keyword in ["task", "action", "activity"]
            ):
                current_section = "tasks"
            elif any(
                keyword in line.lower()
                for keyword in ["milestone", "deliverable", "goal"]
            ):
                current_section = "milestones"
            if line and (line[0].isdigit() or line.startswith(("-", "•", "*"))):
                if current_section == "phases":
                    phases.append(line)
                elif current_section == "tasks":
                    tasks.append(line)
                elif current_section == "milestones":
                    milestones.append(line)
                else:
                    tasks.append(line)
        task_count = len(tasks)
        timeline_estimate = ""
        if time_horizon == "short":
            timeline_estimate = f"{task_count} tasks over 2-4 weeks"
        elif time_horizon == "medium":
            timeline_estimate = f"{task_count} tasks over 2-6 months"
        else:
            timeline_estimate = f"{task_count} tasks over 6-18 months"
        feasibility = 0.7
        if task_count <= 10:
            feasibility += 0.1
        if len(phases) >= 2:
            feasibility += 0.1
        feasibility = min(feasibility, 0.9)
        return {
            "tasks": tasks[:15],
            "phases": phases[:8],
            "milestones": milestones[:10],
            "task_count": len(tasks),
            "phase_count": len(phases),
            "timeline": timeline_estimate,
            "feasibility": feasibility,
            "complexity_assessment": "high"
            if task_count > 15
            else "medium"
            if task_count > 8
            else "low",
            "full_plan": planning_text,
            "task_type": task_type,
        }


class CoordinationAgent(BaseAgent):
    async def execute(self, task: AgentTask, state: GraphState) -> NodeResult:
        """Execute coordination tasks to manage and synchronize multi-agent workflows"""
        try:
            logger.info(
                "CoordinationAgent executing",
                task_id=task.task_id,
                task_type=task.task_type,
            )
            coordination_objective = task.input_data.get("objective", "")
            agent_results = task.input_data.get("agent_results", {})
            workflow_status = task.input_data.get("workflow_status", {})
            coordination_type = task.input_data.get("coordination_type", "general")
            workflow_analysis = self._analyze_workflow_state(
                agent_results, workflow_status
            )
            coordination_plan = await self._generate_coordination_plan(
                coordination_objective, workflow_analysis, coordination_type
            )
            next_actions = self._determine_next_actions(
                coordination_plan, workflow_analysis
            )
            logger.info(
                "Coordination task completed successfully",
                task_id=task.task_id,
                coordination_type=coordination_type,
                next_actions_count=len(next_actions),
            )
            return NodeResult(
                success=True,
                data={
                    "coordination_results": {
                        "workflow_analysis": workflow_analysis,
                        "coordination_plan": coordination_plan,
                        "next_actions": next_actions,
                        "workflow_health": workflow_analysis.get("health_score", 0.7),
                        "completion_estimate": workflow_analysis.get(
                            "completion_estimate", "unknown"
                        ),
                        "bottlenecks": workflow_analysis.get("bottlenecks", []),
                        "recommendations": coordination_plan.get("recommendations", []),
                    },
                    "coordination_type": coordination_type,
                    "agents_coordinated": len(agent_results),
                    "workflow_efficiency": workflow_analysis.get(
                        "efficiency_score", 0.7
                    ),
                },
                confidence=0.8,
                cost=0.01,
            )
        except Exception as e:
            logger.error(
                "CoordinationAgent execution failed", task_id=task.task_id, error=str(e)
            )
            return NodeResult(
                success=False,
                error=f"Coordination agent failed: {str(e)}",
                confidence=0.0,
            )

    def _analyze_workflow_state(
        self, agent_results: dict, workflow_status: dict
    ) -> dict:
        total_agents = len(agent_results)
        completed_agents = sum(
            1
            for result in agent_results.values()
            if result.get("status") == "completed"
        )
        failed_agents = sum(
            1 for result in agent_results.values() if result.get("status") == "failed"
        )
        in_progress_agents = sum(
            1 for result in agent_results.values() if result.get("status") == "working"
        )
        if total_agents > 0:
            completion_rate = completed_agents / total_agents
            failure_rate = failed_agents / total_agents
            health_score = completion_rate - (failure_rate * 0.5)
        else:
            completion_rate = 0.0
            failure_rate = 0.0
            health_score = 0.5
        bottlenecks = []
        for agent_id, result in agent_results.items():
            if (
                result.get("status") == "blocked"
                or result.get("execution_time", 0) > 300
            ):
                bottlenecks.append(agent_id)
        avg_execution_time = sum(
            result.get("execution_time", 0) for result in agent_results.values()
        ) / max(total_agents, 1)
        efficiency_score = 1.0 - min(avg_execution_time / 180, 1.0)
        if completion_rate >= 1.0:
            completion_estimate = "completed"
        elif completion_rate >= 0.8:
            completion_estimate = "near_completion"
        elif completion_rate >= 0.5:
            completion_estimate = "in_progress"
        elif completion_rate > 0:
            completion_estimate = "early_stage"
        else:
            completion_estimate = "not_started"
        return {
            "total_agents": total_agents,
            "completed_agents": completed_agents,
            "failed_agents": failed_agents,
            "in_progress_agents": in_progress_agents,
            "completion_rate": completion_rate,
            "failure_rate": failure_rate,
            "health_score": max(0.0, min(health_score, 1.0)),
            "efficiency_score": max(0.0, min(efficiency_score, 1.0)),
            "bottlenecks": bottlenecks,
            "completion_estimate": completion_estimate,
            "avg_execution_time": avg_execution_time,
        }

    async def _generate_coordination_plan(
        self, objective: str, workflow_analysis: dict, coordination_type: str
    ) -> dict:
        try:
            prompt = self._build_coordination_prompt(
                objective, workflow_analysis, coordination_type
            )
            from app.models.manager import QualityLevel, TaskType

            model_name = self.model_manager.select_optimal_model(
                TaskType.ANALYTICAL_REASONING, QualityLevel.MINIMAL
            )
            result = await self.model_manager.generate(
                model_name=model_name, prompt=prompt, max_tokens=300, temperature=0.2
            )
            if result.success:
                return self._parse_coordination_plan(result.text)
            else:
                return self._generate_fallback_plan(workflow_analysis)
        except Exception as e:
            logger.warning(f"AI coordination planning failed: {e}, using fallback")
            return self._generate_fallback_plan(workflow_analysis)

    def _build_coordination_prompt(
        self, objective: str, workflow_analysis: dict, coordination_type: str
    ) -> str:
        health_score = workflow_analysis.get("health_score", 0.7)
        completion_rate = workflow_analysis.get("completion_rate", 0.0)
        bottlenecks = workflow_analysis.get("bottlenecks", [])
        return f"""You are a workflow coordination expert. Analyze this multi-agent workflow and provide coordination guidance.\n\nObjective: {objective}\nCoordination Type: {coordination_type}\n\nCurrent Workflow Status:\n- Health Score: {health_score:.2f}\n- Completion Rate: {completion_rate:.2f}\n- Active Bottlenecks: {len(bottlenecks)}\n- Completion Estimate: {workflow_analysis.get("completion_estimate", "unknown")}\n\nPlease provide:\n1. Workflow assessment (Good/Fair/Poor)\n2. Top 3 recommendations for improvement\n3. Priority actions to take next\n4. Risk mitigation strategies\n5. Estimated time to completion\n\nKeep recommendations specific and actionable."""

    def _parse_coordination_plan(self, plan_text: str) -> dict:
        recommendations = []
        priority_actions = []
        lines = plan_text.split("\n")
        current_section = None
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if any(
                keyword in line.lower()
                for keyword in ["recommend", "suggest", "should"]
            ):
                current_section = "recommendations"
            elif any(
                keyword in line.lower() for keyword in ["priority", "action", "next"]
            ):
                current_section = "actions"
            if line and (line[0].isdigit() or line.startswith(("-", "•", "*"))):
                if current_section == "recommendations":
                    recommendations.append(line)
                elif current_section == "actions":
                    priority_actions.append(line)
        return {
            "recommendations": recommendations[:5],
            "priority_actions": priority_actions[:3],
            "plan_source": "ai_generated",
            "full_plan": plan_text,
        }

    def _generate_fallback_plan(self, workflow_analysis: dict) -> dict:
        recommendations = []
        priority_actions = []
        health_score = workflow_analysis.get("health_score", 0.7)
        bottlenecks = workflow_analysis.get("bottlenecks", [])
        failure_rate = workflow_analysis.get("failure_rate", 0.0)
        if health_score < 0.5:
            recommendations.append(
                "Workflow health is poor - review failed agents and retry"
            )
            priority_actions.append("Investigate and restart failed agents")
        elif health_score < 0.7:
            recommendations.append("Workflow health is fair - monitor progress closely")
        if bottlenecks:
            recommendations.append(f"Address {len(bottlenecks)} identified bottlenecks")
            priority_actions.append("Resolve bottlenecked agents first")
        if failure_rate > 0.2:
            recommendations.append(
                "High failure rate detected - review agent configurations"
            )
            priority_actions.append("Analyze failure patterns and adjust parameters")
        if not recommendations:
            recommendations.append("Workflow appears healthy - continue monitoring")
            priority_actions.append("Monitor agent progress and completion")
        return {
            "recommendations": recommendations,
            "priority_actions": priority_actions,
            "plan_source": "rule_based_fallback",
            "full_plan": "Generated using rule-based fallback coordination logic",
        }

    def _determine_next_actions(
        self, coordination_plan: dict, workflow_analysis: dict
    ) -> list:
        next_actions = []
        priority_actions = coordination_plan.get("priority_actions", [])
        for action in priority_actions:
            next_actions.append(
                {
                    "action": action,
                    "priority": "high",
                    "type": "coordination",
                    "estimated_effort": "low",
                }
            )
        bottlenecks = workflow_analysis.get("bottlenecks", [])
        if bottlenecks:
            next_actions.append(
                {
                    "action": f"Resolve bottlenecks in agents: {', '.join(bottlenecks)}",
                    "priority": "high",
                    "type": "bottleneck_resolution",
                    "estimated_effort": "medium",
                }
            )
        failed_agents = workflow_analysis.get("failed_agents", 0)
        if failed_agents > 0:
            next_actions.append(
                {
                    "action": f"Retry or reconfigure {failed_agents} failed agents",
                    "priority": "medium",
                    "type": "failure_recovery",
                    "estimated_effort": "medium",
                }
            )
        completion_rate = workflow_analysis.get("completion_rate", 0.0)
        if completion_rate >= 0.8:
            next_actions.append(
                {
                    "action": "Prepare for workflow completion and result synthesis",
                    "priority": "medium",
                    "type": "completion_preparation",
                    "estimated_effort": "low",
                }
            )
        if not next_actions:
            next_actions.append(
                {
                    "action": "Continue monitoring workflow progress",
                    "priority": "low",
                    "type": "monitoring",
                    "estimated_effort": "low",
                }
            )
        return next_actions[:5]


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

    async def execute_tasks(
        self, tasks: List[AgentTask], state: Optional[GraphState] = None
    ) -> Dict[str, NodeResult]:
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
                task
                for task in pending_tasks.values()
                if task.is_ready(completed_tasks)
                and task.status in {AgentStatus.IDLE, AgentStatus.WAITING}
            ]
            if not ready_tasks:
                logger.warning(
                    "No ready tasks found, possible circular dependency or all tasks blocked."
                )
                break

            # Run all ready tasks in parallel
            task_futures = {}
            for task in ready_tasks:
                agent = self.create_agent(task.agent_type)
                task.update_status(AgentStatus.WORKING)
                task_futures[task.task_id] = asyncio.create_task(
                    agent.execute(task, state)
                )

            finished, _ = await asyncio.wait(
                task_futures.values(), return_when=asyncio.ALL_COMPLETED
            )

            # Collect results
            for task_id, future in task_futures.items():
                task = pending_tasks[task_id]
                try:
                    result = future.result()
                except Exception as e:
                    logger.error(
                        "Agent execution failed", task_id=task_id, error=str(e)
                    )
                    task.update_status(AgentStatus.FAILED)
                    if task.can_retry():
                        task.retry_count += 1
                        task.update_status(AgentStatus.WAITING)
                        logger.info(
                            "Retrying task",
                            task_id=task_id,
                            retry_count=task.retry_count,
                        )
                        continue
                    else:
                        results[task_id] = NodeResult(
                            success=False, error=str(e), confidence=0.0
                        )
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
                        logger.info(
                            "Retrying failed task",
                            task_id=task_id,
                            retry_count=task.retry_count,
                        )
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
            max_retries=max_retries,
        )

    async def run_research_workflow(
        self, research_plan: List[Dict[str, Any]], user_context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Orchestrates a multi-agent research workflow based on a research plan.
        Each step can be handled by a different agent, with dependencies and context sharing.
        """
        logger.info("Starting multi-agent research workflow", plan=research_plan)
        # Map user_context fields to GraphState fields if provided
        user_context = user_context or {}
        state = GraphState(
            user_preferences=user_context.get("user_preferences", {}),
            session_id=user_context.get("session_id"),
            cost_budget_remaining=user_context.get("cost_budget", 20.0),
            max_cost=user_context.get("cost_budget"),
            max_execution_time=user_context.get("time_budget", 30.0),
            quality_requirement=user_context.get("quality_requirement", "balanced"),
        )
        tasks = []
        task_id_map = {}
        # Build AgentTasks from research plan
        for step in research_plan:
            agent_type = step["agent_type"]
            task_type = step.get("task_type", "research")
            description = step.get("description", "")
            input_data = step.get("input_data", {})
            dependencies = [
                task_id_map[dep]
                for dep in step.get("depends_on", [])
                if dep in task_id_map
            ]
            task = self.build_task(
                agent_type=agent_type,
                task_type=task_type,
                description=description,
                input_data=input_data,
                dependencies=dependencies,
            )
            tasks.append(task)
            task_id_map[step["id"]] = task.task_id
        # Execute all tasks with dependency resolution
        results = await self.execute_tasks(tasks, state)
        # Aggregate results
        aggregated = {
            "results": {tid: res.dict() for tid, res in results.items()},
            "success": all(res.success for res in results.values()),
            "errors": [res.error for res in results.values() if not res.success],
        }
        logger.info("Research workflow complete", success=aggregated["success"])
        return aggregated

    async def execute_research_workflow(
        self,
        research_question: str,
        methodology: str = "systematic",
        constraints: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Convenience method to build and run a standard multi-agent research workflow.
        """
        constraints = constraints or {}
        workflow_id = str(uuid.uuid4())
        logger.info(
            "Starting research workflow",
            question=research_question,
            methodology=methodology,
            workflow_id=workflow_id,
        )
        # Build the standard research plan steps
        research_plan = [
            {
                "id": "planning",
                "agent_type": AgentType.PLANNING_AGENT,
                "task_type": "research_planning",
                "description": f"Create research plan for: {research_question}",
                "input_data": {
                    "research_question": research_question,
                    "methodology": methodology,
                    "constraints": constraints,
                },
                "depends_on": [],
            },
            {
                "id": "research",
                "agent_type": AgentType.RESEARCH_AGENT,
                "task_type": "information_gathering",
                "description": "Gather comprehensive information",
                "input_data": {"research_plan": "planning_output"},
                "depends_on": ["planning"],
            },
            {
                "id": "analysis",
                "agent_type": AgentType.ANALYSIS_AGENT,
                "task_type": "data_analysis",
                "description": "Analyze research findings",
                "input_data": {"research_data": "research_output"},
                "depends_on": ["research"],
            },
            {
                "id": "fact_check",
                "agent_type": AgentType.FACT_CHECK_AGENT,
                "task_type": "fact_verification",
                "description": "Verify critical claims and sources",
                "input_data": {"analysis_results": "analysis_output"},
                "depends_on": ["analysis"],
            },
            {
                "id": "synthesis",
                "agent_type": AgentType.SYNTHESIS_AGENT,
                "task_type": "report_generation",
                "description": "Generate comprehensive research report",
                "input_data": {
                    "all_findings": "multiple_outputs",
                    "research_question": research_question,
                },
                "depends_on": ["fact_check"],
            },
        ]
        # Run the workflow using the generic method
        return await self.run_research_workflow(research_plan)
