import re

from agent.tools import CONCEPTS_DB, SETUP_GUIDES_DB


def test_concept_minimum_python_versions_match_installation_guide():
    for concept in CONCEPTS_DB.values():
        version = concept["min_python"].removesuffix("+")
        assert tuple(map(int, version.split("."))) >= (3, 9)

    match = re.search(
        r"Minimum supported Python is (?P<version>\d+\.\d+) for langchain/langgraph",
        SETUP_GUIDES_DB["installation"],
    )
    assert match is not None
    assert CONCEPTS_DB["langchain"]["min_python"] == f"{match['version']}+"
    assert CONCEPTS_DB["langgraph"]["min_python"] == f"{match['version']}+"
