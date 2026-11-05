# Mol-1 Constraint

`mol-1-constraint` adds SITE, CONTACT and EXCLUSION conditioning to Mol-1.
The released model supports all three; each input may provide any combination
or none. Conditioning guides predictions without guaranteeing satisfaction.

- [Prediction and weight download](../../README.md#predict-with-mol-1-constraint)
- [Constraint semantics and input example](../../docs/constraint_semantics.md)
- [Evidence-to-constraints skill](../../docs/mol1-evidence-to-constraints/SKILL.md)
- [Prepared 7PZB example](../../examples/7pzb_constraint_binding_site/input.json)

[`config.py`](config.py) holds inference defaults.
No configuration file is required; `--config` JSON overrides remain supported.
