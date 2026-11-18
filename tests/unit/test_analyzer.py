from control_panel.orchestrator.analyzer import _extract_task_spec_json

def test_extract_fenced_valid_payload():
    """1. fenced valid payload -> succeeds"""
    text = '```json\n{"objective": "test", "subtasks": [], "category": "test"}\n```'
    res = _extract_task_spec_json(text)
    assert res is not None
    assert res["objective"] == "test"

def test_extract_unfenced_json_surrounded_by_prose():
    """2. unfenced JSON surrounded by prose -> succeeds"""
    text = 'Here is the payload:\n{"objective": "test", "subtasks": [], "category": "test"}\nHope it helps!'
    res = _extract_task_spec_json(text)
    assert res is not None
    assert res["objective"] == "test"

def test_extract_skips_invalid_first_object():
    """3. first JSON lacks required fields, second is valid -> second selected"""
    text = '{"thinking": "First I do this"}\nThen output:\n{"objective": "test", "subtasks": [], "category": "test"}'
    res = _extract_task_spec_json(text)
    assert res is not None
    assert "objective" in res
    assert "thinking" not in res

def test_extract_multiple_fenced_blocks():
    """4. multiple fenced JSON blocks -> correct payload selected"""
    text = '```json\n{"thinking": "blah"}\n```\n```json\n{"objective": "test", "subtasks": [], "category": "test"}\n```'
    res = _extract_task_spec_json(text)
    assert res is not None
    assert "objective" in res

def test_extract_nested_objects():
    """5. nested objects/arrays inside subtasks -> succeeds"""
    text = '{"objective": "test", "subtasks": [{"name": "scan"}], "category": "test"}'
    res = _extract_task_spec_json(text)
    assert res is not None
    assert res["subtasks"][0]["name"] == "scan"

def test_extract_braces_inside_strings():
    """6. braces inside JSON strings -> succeeds"""
    text = '{"objective": "this {brace} is fine", "subtasks": [], "category": "test"}'
    res = _extract_task_spec_json(text)
    assert res is not None
    assert "{" in res["objective"]

def test_extract_truncated_json_rejected():
    """7. truncated JSON -> rejected"""
    text = '{"objective": "test", "subtasks": [], "category": "te'
    res = _extract_task_spec_json(text)
    assert res is None

def test_extract_malformed_trailing_comma():
    """8. trailing-comma malformed JSON -> rejected"""
    text = '{"objective": "test", "subtasks": [], "category": "test",}'
    res = _extract_task_spec_json(text)
    assert res is None

def test_extract_empty_or_non_json():
    """9. empty/non-JSON response -> rejected"""
    text = "I cannot do that."
    res = _extract_task_spec_json(text)
    assert res is None

def test_extract_missing_objective():
    """10. valid JSON missing objective -> rejected"""
    text = '{"subtasks": [], "category": "test"}'
    res = _extract_task_spec_json(text)
    assert res is None

def test_extract_missing_subtasks():
    """11. valid JSON missing subtasks -> rejected"""
    text = '{"objective": "test", "category": "test"}'
    res = _extract_task_spec_json(text)
    assert res is None

def test_extract_missing_category():
    """12. valid JSON missing category -> rejected"""
    text = '{"objective": "test", "subtasks": []}'
    res = _extract_task_spec_json(text)
    assert res is None

def test_extract_thinking_alone_rejected():
    """13. {"thinking": "..."} alone -> rejected and cannot silently become a successful GenAI analysis"""
    text = '{"thinking": "I should organize files"}'
    res = _extract_task_spec_json(text)
    assert res is None
