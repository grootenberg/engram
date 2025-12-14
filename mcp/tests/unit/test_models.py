"""Unit tests for data models."""

from datetime import datetime, timedelta

import pytest

from app.models.memory import Memory, MemoryType, ObservationType, SectionType
from app.models.reflection_state import ReflectionState


class TestMemoryModel:
    """Tests for the Memory model."""

    def test_effective_importance_base(self, sample_memory):
        """Test effective importance with no feedback."""
        sample_memory.helpful_count = 0
        sample_memory.harmful_count = 0
        assert sample_memory.effective_importance == sample_memory.importance_score

    def test_effective_importance_with_helpful(self, sample_memory):
        """Test effective importance increases with helpful feedback."""
        sample_memory.importance_score = 5.0
        sample_memory.helpful_count = 4
        sample_memory.harmful_count = 0
        # 5.0 + (4 - 0) * 0.5 = 7.0
        assert sample_memory.effective_importance == 7.0

    def test_effective_importance_with_harmful(self, sample_memory):
        """Test effective importance decreases with harmful feedback."""
        sample_memory.importance_score = 5.0
        sample_memory.helpful_count = 0
        sample_memory.harmful_count = 4
        # 5.0 + (0 - 4) * 0.5 = 3.0
        assert sample_memory.effective_importance == 3.0

    def test_effective_importance_clamped_max(self, sample_memory):
        """Test effective importance is clamped to max 10."""
        sample_memory.importance_score = 9.0
        sample_memory.helpful_count = 10
        sample_memory.harmful_count = 0
        # Would be 14.0, but clamped to 10.0
        assert sample_memory.effective_importance == 10.0

    def test_effective_importance_clamped_min(self, sample_memory):
        """Test effective importance is clamped to min 1."""
        sample_memory.importance_score = 2.0
        sample_memory.helpful_count = 0
        sample_memory.harmful_count = 10
        # Would be -3.0, but clamped to 1.0
        assert sample_memory.effective_importance == 1.0

    def test_to_dict_serialization(self, sample_memory):
        """Test memory serialization to dict."""
        result = sample_memory.to_dict()

        assert result["content"] == sample_memory.content
        assert result["memory_type"] == sample_memory.memory_type.value
        assert result["importance_score"] == sample_memory.importance_score
        assert "effective_importance" in result
        assert "created_at" in result

    def test_memory_type_enum(self):
        """Test MemoryType enum values."""
        assert MemoryType.EPISODIC.value == "episodic"
        assert MemoryType.SEMANTIC.value == "semantic"
        assert MemoryType.PROCEDURAL.value == "procedural"

    def test_observation_type_enum(self):
        """Test ObservationType enum values."""
        assert ObservationType.ERROR.value == "error"
        assert ObservationType.INSTRUCTION.value == "instruction"
        assert ObservationType.DECISION.value == "decision"
        assert ObservationType.CODE_CHANGE.value == "code_change"

    def test_section_type_enum(self):
        """Test SectionType enum values."""
        assert SectionType.STRATEGIES.value == "strategies"
        assert SectionType.SNIPPETS.value == "snippets"
        assert SectionType.PITFALLS.value == "pitfalls"
        assert SectionType.CONTEXT.value == "context"
        assert SectionType.PREFERENCES.value == "preferences"


class TestReflectionStateModel:
    """Tests for the ReflectionState model."""

    def test_should_reflect_importance_threshold(self, test_user_id):
        """Test reflection triggers on importance threshold."""
        state = ReflectionState(
            user_id=test_user_id,
            accumulated_importance=160.0,
            observations_since=50,
            last_reflection_at=datetime.utcnow(),
        )
        should, reason = state.should_reflect(importance_threshold=150.0)
        assert should is True
        assert "accumulated_importance" in reason

    def test_should_reflect_observation_threshold(self, test_user_id):
        """Test reflection triggers on observation count."""
        state = ReflectionState(
            user_id=test_user_id,
            accumulated_importance=50.0,
            observations_since=110,
            last_reflection_at=datetime.utcnow(),
        )
        should, reason = state.should_reflect(observation_threshold=100)
        assert should is True
        assert "observations_since" in reason

    def test_should_reflect_time_threshold(self, test_user_id):
        """Test reflection triggers on time threshold."""
        state = ReflectionState(
            user_id=test_user_id,
            accumulated_importance=50.0,
            observations_since=10,
            last_reflection_at=datetime.utcnow() - timedelta(hours=30),
        )
        should, reason = state.should_reflect(time_threshold_hours=24)
        assert should is True
        assert "hours_since_last" in reason

    def test_should_not_reflect(self, test_user_id):
        """Test reflection does not trigger when thresholds not met."""
        state = ReflectionState(
            user_id=test_user_id,
            accumulated_importance=50.0,
            observations_since=10,
            last_reflection_at=datetime.utcnow(),
        )
        should, reason = state.should_reflect()
        assert should is False
        assert "no trigger conditions met" in reason

    def test_accumulate(self, test_user_id):
        """Test accumulating importance."""
        state = ReflectionState(
            user_id=test_user_id,
            accumulated_importance=0.0,
            observations_since=0,
        )
        state.accumulate(5.0)
        state.accumulate(7.0)

        assert state.accumulated_importance == 12.0
        assert state.observations_since == 2

    def test_reset(self, test_user_id):
        """Test resetting state after reflection."""
        state = ReflectionState(
            user_id=test_user_id,
            accumulated_importance=160.0,
            observations_since=110,
            last_reflection_at=datetime.utcnow() - timedelta(hours=48),
        )
        state.reset()

        assert state.accumulated_importance == 0.0
        assert state.observations_since == 0
        # last_reflection_at should be recent
        assert (datetime.utcnow() - state.last_reflection_at).total_seconds() < 5

    def test_to_dict(self, test_user_id):
        """Test serialization to dict."""
        state = ReflectionState(
            user_id=test_user_id,
            accumulated_importance=75.0,
            observations_since=30,
        )
        result = state.to_dict()

        assert result["accumulated_importance"] == 75.0
        assert result["observations_since"] == 30
        assert "hours_since_last" in result
        assert "last_reflection_at" in result
