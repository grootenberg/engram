# Comprehensive Code Review Report: Engram (memocc)

**Date**: 2025-12-09
**Version**: v1.0.0
**Review Type**: Full Multi-Dimensional Analysis

---

## Executive Summary

| Dimension     | Score    | Status            |
| ------------- | -------- | ----------------- |
| Code Quality  | 7.5/10   | Needs Improvement |
| Architecture  | B+       | Strong            |
| Security      | CRITICAL | Block Deployment  |
| Test Coverage | 60%      | Gaps Identified   |

**Overall Recommendation**: **DO NOT deploy to production** until Critical (P0) security issues are resolved.

---

## Issue Summary by Priority

| Priority          | Count | Categories                                     |
| ----------------- | ----- | ---------------------------------------------- |
| **P0 - Critical** | 4     | SQL Injection, No Auth, Hardcoded Creds, IDOR  |
| **P1 - High**     | 5     | Missing Tests, Arch Violations, Error Handling |
| **P2 - Medium**   | 6     | Code Duplication, Complexity, Design Patterns  |
| **P3 - Low**      | 4     | Documentation, Magic Numbers, Style            |

---

## P0 - Critical Issues (Fix Immediately)

### SEC-001: SQL Injection Vulnerability

**CVSS Score**: 9.8 (Critical)
**Files**: `app/services/ingestion.py`, `app/services/retrieval.py`, `app/services/reflection.py`

**Problem**: Raw SQL queries using `text()` with string interpolation in type/synthetic filters:

```python
# retrieval.py:68-69
types_str = ", ".join(f"'{t.upper()}'" for t in memory_types)
type_filter = f"AND memory_type::text IN ({types_str})"
```

**Impact**: Attacker can execute arbitrary SQL commands, exfiltrate data, or destroy database.

**Remediation**:

```python
# Use parameterized queries with ANY()
type_filter = "AND memory_type::text = ANY(:memory_types)"
params["memory_types"] = [t.upper() for t in memory_types]
```

---

### SEC-002: No Authentication/Authorization

**CVSS Score**: 9.1 (Critical)
**Files**: All MCP tools (`app/tools/*.py`)

**Problem**: No authentication mechanism. Any client can call tools. User ID accepted as parameter without verification.

**Impact**: Complete system compromise. Any user can access/modify any other user's memories.

**Remediation**:

1. Implement JWT/API key authentication at MCP server level
2. Extract user identity from authenticated session, not parameters
3. Add authorization middleware to validate user access

---

### SEC-003: Hardcoded Database Credentials

**CVSS Score**: 7.5 (High)
**File**: `app/config.py:16`

**Problem**:

```python
database_url: str = "postgresql+asyncpg://postgres:birdseye@localhost:5432/engram"
```

**Impact**: Credentials exposed in source code, version control, and any code analysis.

**Remediation**:

```python
database_url: str = Field(default=None)  # Require explicit environment variable

@validator("database_url", pre=True)
def validate_database_url(cls, v):
    if not v:
        raise ValueError("DATABASE_URL environment variable is required")
    return v
```

---

### SEC-004: Insecure Direct Object Reference (IDOR)

**CVSS Score**: 8.6 (High)
**Files**: All MCP tools

**Problem**: `user_id` parameter accepted without verification. Any caller can specify any user ID.

**Impact**: User impersonation, unauthorized data access across all users.

**Remediation**:

```python
# Derive user_id from authenticated context, not parameters
@mcp.tool()
async def memory_observe(content: str, ...):  # Remove user_id parameter
    user_id = get_authenticated_user_id()  # From session/token
```

---

## P1 - High Priority Issues

### TEST-001: ReflectionService Has Zero Tests

**File**: `app/services/reflection.py` (282 lines)

**Problem**: Critical service with LLM integration, database operations, and complex logic has no test coverage.

**Impact**: Regression risk, silent failures, untested edge cases.

**Remediation**: Add tests for:

- `should_reflect()` trigger logic
- `reflect()` happy path and error cases
- `_get_reflection_candidates()` with/without focus
- LLM response parsing edge cases
- Citation validation

---

### TEST-002: Stats Tool Has Zero Tests

**File**: `app/tools/stats.py`

**Problem**: Tool that queries database and returns system metrics is completely untested.

**Remediation**: Add tests for all scope types (summary, detailed, reflection).

---

### ARCH-001: Tools Bypassing Service Layer

**Files**: `app/tools/feedback.py`, `app/tools/stats.py`

**Problem**: Direct database access in tool layer violates layered architecture.

**Impact**: Business logic scattered across layers, harder to test and maintain.

**Remediation**: Create `FeedbackService` and `StatsService` classes.

---

### CODE-001: Generic Exception Handling

**File**: `app/db/connection.py`

**Problem**: Catching `Exception` broadly hides specific errors.

**Remediation**: Catch specific exceptions (`asyncpg.PostgresError`, `SQLAlchemyError`).

---

### CODE-002: No Retry/Circuit Breaker for External Services

**Files**: `app/services/embedding.py`, `app/services/reflection.py`

**Problem**: External API calls (OpenAI, Anthropic) have no resilience patterns.

**Impact**: Transient failures cause hard errors, no graceful degradation.

**Remediation**: Add `tenacity` retry decorator with exponential backoff.

---

## P2 - Medium Priority Issues

### CODE-003: User ID Parsing Duplication

**Files**: All 5 tool files (`app/tools/*.py`)

**Problem**: Identical 6-line code block repeated in every tool:

```python
try:
    uid = UUID(user_id)
except ValueError:
    import hashlib
    uid = UUID(hashlib.md5(user_id.encode()).hexdigest())
```

**Remediation**: Extract to `app/utils/user.py`:

```python
def parse_user_id(user_id: str) -> UUID:
    """Parse user ID string to UUID, generating deterministic UUID for non-UUID strings."""
    try:
        return UUID(user_id)
    except ValueError:
        return UUID(hashlib.md5(user_id.encode()).hexdigest())
```

---

### CODE-004: Long Method in RetrievalService

**File**: `app/services/retrieval.py` (158 lines in `retrieve()`)

**Problem**: Single method handles weight normalization, query building, SQL execution, result formatting.

**Remediation**: Extract into focused methods:

- `_normalize_weights()`
- `_build_query_filters()`
- `_execute_retrieval_query()`
- `_format_results()`

---

### CODE-005: Mock Class Anti-Pattern

**File**: `app/services/ingestion.py:148-153`

**Problem**: Inline class definition in method:

```python
class DuplicateResult:
    id = row.id
    similarity = row.similarity
return DuplicateResult()
```

**Remediation**: Use `dataclass` or `NamedTuple`:

```python
from dataclasses import dataclass

@dataclass
class DuplicateResult:
    id: UUID
    similarity: float
```

---

### ARCH-002: Missing Repository Pattern

**Impact**: Services directly use SQLAlchemy, mixing business and data access logic.

**Remediation**: Add `app/repositories/` layer:

```python
class MemoryRepository:
    async def find_by_user(self, user_id: UUID) -> list[Memory]: ...
    async def find_similar(self, embedding: list[float], threshold: float) -> Memory | None: ...
```

---

### ARCH-003: Global Singleton Anti-Pattern

**Files**: All service modules

**Problem**: Global instances (`embedding_service = EmbeddingService()`) make testing difficult.

**Remediation**: Use dependency injection:

```python
from functools import lru_cache

@lru_cache()
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService()
```

---

### ARCH-004: Missing Dependency Injection Framework

**Impact**: Hard-coded dependencies reduce testability and flexibility.

**Recommendation**: Consider `fastapi.Depends` pattern or `dependency-injector` library.

---

## P3 - Low Priority Issues

### DOC-001: Magic Numbers Without Documentation

**File**: `app/config.py`

**Problem**: Configuration values lack explanatory comments:

```python
engram_recency_decay: float = 0.995
engram_importance_threshold: float = 150.0
```

**Remediation**: Add docstrings explaining each parameter's purpose and valid ranges.

---

### DOC-002: Missing Docstrings in Models

**Files**: `app/models/memory.py`, `app/models/reflection_state.py`

**Remediation**: Add class and field docstrings.

---

### STYLE-001: Inconsistent Import Organization

**Files**: Various

**Remediation**: Use `isort` with consistent profile.

---

### STYLE-002: Long Lines in SQL Queries

**Files**: `app/services/retrieval.py`, `app/services/reflection.py`

**Remediation**: Break into multiple lines with clear indentation.

---

## Architecture Assessment

### Strengths

- Clean layer separation: Tools → Services → Database
- No circular dependencies detected
- Consistent async/await patterns throughout
- Well-structured configuration management
- Good use of enums for type safety

### Weaknesses

- No Repository Pattern (services directly use SQLAlchemy)
- No Dependency Injection framework
- Global singletons make testing harder
- Missing resilience patterns (retry, circuit breaker)
- Two tools bypass service layer

### Recommended Architecture Changes

1. Add Repository layer between Services and Database
2. Implement DI container or factory pattern
3. Add resilience middleware for external services
4. Create FeedbackService and StatsService

---

## Test Coverage Assessment

### Current State

- **Total Tests**: 48 (18 unit, 25 integration, 5 eval)
- **Estimated Coverage**: ~60%
- **Infrastructure**: Excellent (Docker, fixtures, factories, VCR)

### Critical Gaps

| Component         | Current | Required |
| ----------------- | ------- | -------- |
| ReflectionService | 0%      | 80%+     |
| Stats Tool        | 0%      | 80%+     |
| Error Handling    | 30%     | 70%+     |
| Edge Cases        | 40%     | 60%+     |

### Test Recommendations

1. Add comprehensive ReflectionService tests (LLM mocking, edge cases)
2. Add Stats tool tests for all scope types
3. Add error handling tests for external service failures
4. Add boundary condition tests for scoring algorithms

---

## Remediation Roadmap

### Phase 1: Security (1-2 days)

1. Replace string interpolation with parameterized queries
2. Remove hardcoded credentials
3. Design authentication architecture

### Phase 2: Critical Tests (2-3 days)

1. Add ReflectionService test suite
2. Add Stats tool tests
3. Add error handling tests

### Phase 3: Architecture (3-5 days)

1. Create FeedbackService and StatsService
2. Extract user ID parsing utility
3. Refactor long methods

### Phase 4: Infrastructure (2-3 days)

1. Add Repository layer
2. Implement retry patterns
3. Add dependency injection

### Phase 5: Polish (1-2 days)

1. Fix documentation gaps
2. Apply style fixes
3. Configure linting rules

---

## Files Changed by This Review

| File                             | Action                      |
| -------------------------------- | --------------------------- |
| `COMPREHENSIVE_REVIEW_REPORT.md` | Created                     |
| `SECURITY_AUDIT_REPORT.md`       | Created (by security agent) |

---

## Conclusion

The Engram codebase demonstrates solid foundational architecture with clean layer separation and consistent patterns. However, **critical security vulnerabilities prevent production deployment**. The SQL injection and authentication gaps must be addressed before any user-facing deployment.

After security fixes, focus on test coverage for ReflectionService and Stats tool to ensure system reliability. The architectural improvements (Repository pattern, DI) are recommended for long-term maintainability but can be deferred.

**Next Steps**:

1. Create Linear issues for P0 and P1 items
2. Block deployment pipeline until SEC-001 through SEC-004 are resolved
3. Schedule security-focused code review after fixes
