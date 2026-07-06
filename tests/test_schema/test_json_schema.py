"""JSON Schema export: the discriminator mapping must be present and must route
both variants, so cross-language consumers can dispatch without Python."""

from rolloutscope.schema import rollout_json_schema


def test_discriminator_mapping_present() -> None:
    schema = rollout_json_schema()
    discriminator = schema["discriminator"]
    assert discriminator["propertyName"] == "kind"
    mapping = discriminator["mapping"]
    assert set(mapping) == {"single_turn", "multi_turn"}
    for ref in mapping.values():
        assert ref.startswith("#/$defs/")
        assert ref.split("/")[-1] in schema["$defs"]


def test_variants_and_version_in_schema() -> None:
    schema = rollout_json_schema()
    assert "oneOf" in schema
    assert len(schema["oneOf"]) == 2
    single = schema["$defs"]["SingleTurnRollout"]
    assert "schema_version" in single["properties"]
    assert single["additionalProperties"] is True
    multi = schema["$defs"]["MultiTurnRollout"]
    assert "trajectory" in multi["properties"]
