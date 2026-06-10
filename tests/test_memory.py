from legal_review_agent.memory.memory_manager import MemoryEntry, MemoryManager


def test_three_tier_memory(tmp_path):
    m = MemoryManager(str(tmp_path), "deal-001")

    m.long_term.add(MemoryEntry(key="bl1", content="违约金不超过20%", tags=["baseline"]))
    m.long_term.add(MemoryEntry(key="bl2", content="黑名单：无限连带责任", tags=["blacklist"]))
    assert len(m.long_term.scan()) == 2
    assert len(m.long_term.scan(tags=["blacklist"])) == 1

    m.session.set_fact("contract_type", "采购合同")
    m.session.record_turn("user", "请审查这份合同")
    assert m.session.facts()["contract_type"] == "采购合同"

    m.working.put("result:parse_clauses", "{...}")
    assert "result:parse_clauses" in m.working.snapshot()


def test_session_memory_persists_across_instances(tmp_path):
    MemoryManager(str(tmp_path), "deal-002").session.set_fact("stance", "强硬")
    reopened = MemoryManager(str(tmp_path), "deal-002")
    assert reopened.session.facts()["stance"] == "强硬"
