"""
Conclusion-verified Analogical Schema Induction (CASI)
Part 4: Schema Extractor

Extracts a generalized, abstract schema from multiple successful
analogical mappings.
"""

from collections import defaultdict

from .schema_mapper import Mapping


class SchemaExtractor:
    """
    Induces an abstract schema from multiple analogous examples.
    This process takes several concrete examples (mappings) and abstracts
    out the common relational structure, replacing specific entities with
    generic placeholders.
    """

    def extract_schema(self, mappings: list[Mapping], base_cases: dict) -> dict:
        """
        Extract a schema by finding the common relational subgraph.
        Args:
            mappings: A list of verified Mapping objects.
            base_cases: A dict of the base case data used in the mappings.
        Returns:
            An abstract schema as a dict.
        """
        if not mappings:
            return {}

        # Find common relations across all base cases involved in the mappings
        base_case_ids = {m.base_case_id for m in mappings}
        first_base_id = list(base_case_ids)[0]
        common_relations = self._get_common_relations(base_case_ids, base_cases)

        if not common_relations:
            return {}

        # Create the abstract schema from the common structure
        schema_relations, entity_map = self._abstract_relations(common_relations)

        schema = {
            "id": f"schema-{first_base_id}",
            "entities": list(entity_map.values()),
            "relations": schema_relations,
            "source_mappings": [m.base_case_id for m in mappings],
        }
        return schema

    def _get_common_relations(self, base_case_ids: set, base_cases: dict) -> list:
        """Find the intersection of relations across all specified base cases."""
        if not base_case_ids:
            return []

        # Get relation signatures (type, num_args) from the first case
        first_case_id = list(base_case_ids)[0]
        first_case_relations = base_cases[first_case_id].get("relations", [])

        # Using a tuple of tuples for relation signature to make it hashable
        relation_signatures = {
            (r["type"], tuple(r["args"])) for r in first_case_relations
        }

        # Intersect with relations from other cases
        for case_id in list(base_case_ids)[1:]:
            case_relations = {
                (r["type"], tuple(r["args"]))
                for r in base_cases[case_id].get("relations", [])
            }
            relation_signatures.intersection_update(case_relations)

        return [
            {"type": sig[0], "args": list(sig[1])} for sig in relation_signatures
        ]

    def _abstract_relations(self, relations: list) -> tuple[list, dict]:
        """Replace concrete entity names with generic placeholders."""
        entity_map = {}
        placeholder_idx = 0

        def get_placeholder(entity_name):
            nonlocal placeholder_idx
            if entity_name not in entity_map:
                placeholder_idx += 1
                entity_map[entity_name] = f"?VAR{placeholder_idx}"
            return entity_map[entity_name]

        abstract_relations = []
        for rel in relations:
            abstract_args = [get_placeholder(arg) for arg in rel["args"]]
            abstract_relations.append({"type": rel["type"], "args": abstract_args})

        return abstract_relations, entity_map
