import re

from agent.tools import CONCEPTS_DB, SETUP_GUIDES_DB


def test_concept_minimum_python_versions_match_installation_guide():
    stated_versions = {
        f"{version}+"
        for version in re.findall(r"\d+\.\d+", SETUP_GUIDES_DB["installation"])
    }

    assert all(data["min_python"] in stated_versions for data in CONCEPTS_DB.values())
