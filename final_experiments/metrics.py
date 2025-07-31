from typing import TypeAlias

Rules: TypeAlias = list[list[list[tuple[int, tuple[float, float]]]]]


def get_rule_size(rules: Rules):
    return sum(len(class_rules) for class_rules in rules)


def get_literal_count(rules: Rules):
    return sum(
        len(conjunction)
        for class_rule in rules
        for clause in class_rule
        for conjunction in clause
    )
