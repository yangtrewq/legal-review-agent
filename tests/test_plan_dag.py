import pytest

from legal_review_agent.types import ExecutionPlan, SubTask


def test_topological_batches_parallel_and_sequential():
    plan = ExecutionPlan(goal="g", tasks=[
        SubTask(id="t1", name="基线拉取", description="", depends_on=[]),
        SubTask(id="t2", name="条款解析", description="", depends_on=[]),
        SubTask(id="t3", name="风控比对", description="", depends_on=["t1", "t2"]),
        SubTask(id="t4", name="意见生成", description="", depends_on=["t3"]),
    ])
    batches = plan.topological_batches()
    assert [sorted(t.id for t in b) for b in batches] == [["t1", "t2"], ["t3"], ["t4"]]


def test_cycle_detection():
    plan = ExecutionPlan(goal="g", tasks=[
        SubTask(id="a", name="a", description="", depends_on=["b"]),
        SubTask(id="b", name="b", description="", depends_on=["a"]),
    ])
    with pytest.raises(ValueError):
        plan.topological_batches()


def test_dangling_dependency():
    plan = ExecutionPlan(goal="g", tasks=[
        SubTask(id="a", name="a", description="", depends_on=["missing"]),
    ])
    with pytest.raises(ValueError):
        plan.topological_batches()
