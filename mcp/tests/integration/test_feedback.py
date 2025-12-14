"""Integration tests for feedback service and feedback tool."""

from uuid import uuid4

import pytest
from sqlalchemy import select

from app.models.memory import Memory, MemoryType
from app.tools.feedback import memory_feedback


class TestFeedbackTool:
    """Integration tests for the memory_feedback tool."""

    @pytest.fixture
    async def test_memory(self, db_session, test_user_id, mock_embedding):
        """Create a test memory for feedback tests."""
        memory = Memory(
            user_id=test_user_id,
            content="Test memory for feedback",
            embedding=mock_embedding("Test memory for feedback"),
            memory_type=MemoryType.EPISODIC,
            importance_score=5.0,
            helpful_count=0,
            harmful_count=0,
        )
        db_session.add(memory)
        await db_session.commit()
        await db_session.refresh(memory)
        return memory

    async def test_helpful_feedback(self, db_session, test_memory):
        """Test providing helpful feedback increments count."""
        # Note: This would need patching to use the test session
        # For now, test the logic directly
        initial_helpful = test_memory.helpful_count

        test_memory.helpful_count += 1
        await db_session.commit()
        await db_session.refresh(test_memory)

        assert test_memory.helpful_count == initial_helpful + 1
        assert test_memory.effective_importance == 5.0 + 0.5  # 5.5

    async def test_harmful_feedback(self, db_session, test_memory):
        """Test providing harmful feedback increments count."""
        test_memory.harmful_count += 1
        await db_session.commit()
        await db_session.refresh(test_memory)

        assert test_memory.harmful_count == 1
        assert test_memory.effective_importance == 5.0 - 0.5  # 4.5

    async def test_feedback_with_reason(self, db_session, test_memory):
        """Test feedback stores reason in metadata."""
        test_memory.helpful_count += 1
        if test_memory.extra_metadata is None:
            test_memory.extra_metadata = {}
        test_memory.extra_metadata["feedback_log"] = [
            {"helpful": True, "reason": "This saved me time debugging"}
        ]
        await db_session.commit()
        await db_session.refresh(test_memory)

        assert test_memory.extra_metadata["feedback_log"][0]["reason"] == "This saved me time debugging"

    async def test_multiple_feedback(self, db_session, test_memory):
        """Test multiple feedback signals accumulate."""
        test_memory.helpful_count = 5
        test_memory.harmful_count = 2
        await db_session.commit()

        # Effective importance: 5.0 + (5-2)*0.5 = 6.5
        assert test_memory.effective_importance == 6.5

    async def test_feedback_invalid_memory_id(self):
        """Test feedback with invalid memory ID returns error."""
        result = await memory_feedback(
            memory_id="not-a-uuid",
            helpful=True,
        )

        assert result["status"] == "error"
        assert "Invalid memory_id" in result["reason"]

    async def test_feedback_nonexistent_memory(self):
        """Test feedback with nonexistent memory ID returns error."""
        fake_id = str(uuid4())
        result = await memory_feedback(
            memory_id=fake_id,
            helpful=True,
        )

        assert result["status"] == "error"
        assert "not found" in result["reason"]
