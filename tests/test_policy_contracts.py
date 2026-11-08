from src.policy.contracts import TOOL_CONTRACTS


def test_tool_contracts_not_empty():
    assert len(TOOL_CONTRACTS) > 0


def test_all_contracts_have_tier():
    for name, contract in TOOL_CONTRACTS.items():
        assert hasattr(contract, "tier"), f"{name} missing tier"


def test_all_contract_names_are_strings():
    for name in TOOL_CONTRACTS:
        assert isinstance(name, str) and len(name) > 0


def test_get_sensor_reading_in_contracts():
    assert "get_sensor_reading" in TOOL_CONTRACTS
