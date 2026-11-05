# Constraint semantics

Mol-1 supports the learned conditioning signals: SITE, CONTACT and EXCLUSION.
They guide structure prediction through updates to the model's pair representation.
They do not guarantee satisfaction: all candidates use the base output pipeline,
without additional constraint scoring or filtering. Inference needs no reference structure.

## SITE: define the binding site

SITE marks **source** protein residues that should *contact* the declared **partner** chain(s),
Each selected residue should contact any member of the combined partner set;
it need not contact every partner chain or a particular partner residue or atom.
The partner scope includes every token in those chains; partners may be proteins,
peptides, ligands, DNA or RNA. A partner set may mix protein and non-protein tokens.

## CONTACT: define the fine-grained contact pairs

CONTACT specifies a pair of endpoints that should have a minimum heavy-atom distance
of at most **d_contact** (by default: = 4.5 Å; fixed checkpoint contract).
Pairs are undirected; reversed and duplicate pairs are deduplicated.
Unmarked pairs are unknown, not negative contacts. Residue-pair input supports
protein residues; token-pair input also supports ligands, DNA and RNA.
Same-chain pairs, including adjacent residues, are allowed in explicit input.

## EXCLUSION: define the forbidden site for binding

EXCLUSION marks source protein residues that should *stay away* from the declared
partner chains, using the same source and partner rules as SITE.
Each selected residue should stay away from every member of the combined partner set.
For each selected residue, the minimum heavy-atom distance to the full partner
should be **strictly greater than d_exclusion** (by default: = 6 Å), where
**d_exclusion = d_contact + d_gap** and **d_gap** (by default: = 1.5 Å) is the gap.
Equality violates this geometric condition, but does not reject a prediction.
This means geometric separation, not absence of physical interaction.
EXCLUSION is not the logical complement of CONTACT because their cutoffs differ.

## Input and numbering

Add any combination of `site`, `contact` and `exclusion` to a native task's
`constraints` object. SITE / EXCLUSION use `source_chains`, `partner_chains` and
`residue_ids`; CONTACT uses `pairs` or `token_pairs`.
Each task accepts a single SITE object and a single EXCLUSION object; within each,
all selected residues share the same partner set. Independent groups are not supported.

For example, add `constraints` to a task as shown below. Other fields are omitted
with `...`; see the [input JSON format](infer_json_format.md) for the full task
structure. This example assumes protein chains A and B contain the listed residues.

```jsonc
[
  {
    ...,
    "constraints": {
      "site": {
        "source_chains": ["A"],
        "partner_chains": ["B"],
        "residue_ids": [["A", 8], ["A", 12]]
      },
      "contact": {
        "pairs": [[["A", 8], ["B", 15]]]
      },
      "exclusion": {
        "source_chains": ["A"],
        "partner_chains": ["B"],
        "residue_ids": [["A", 22]]
      }
    }
  }
]
```

- Active SITE / EXCLUSION inputs require nonempty, unique `source_chains` and
  `partner_chains` lists, with no chain in both roles. Multiple chains are allowed;
  same-chain SITE / EXCLUSION is not supported.
- `residue_ids` and CONTACT `pairs` use `[chain_id, position]`. Positions are full-sequence
  `label_seq_id` values with base **b_residue** (by default: = 1; fixed input convention),
  not PDB author numbering.
  Selected residues must map to protein tokens; SITE / EXCLUSION residues must belong to source chains.
  Inline SITE / EXCLUSION residue lists are sorted and deduplicated automatically.
- Instead of `pairs`, CONTACT accepts `token_pairs` containing token-index pairs.
  Indices use base **b_token** (by default: = 0; fixed input convention)
  on the **complete featurized token axis**, not chain-local
  positions or general atom indices. Do not combine `pairs` and `token_pairs`.
- Active constraints reject missing chains or residues after preprocessing,
  out-of-range token indices and identical CONTACT endpoints. Unsupported fields are always errors.
  Empty SITE / EXCLUSION residue lists deactivate the condition before chain-role or existence checks.
- Omitted, `null` or `{}` constraints are inactive. An empty `residue_ids`, `pairs`
  or `token_pairs` list also supplies no condition. A nonempty SITE / EXCLUSION
  object without `residue_ids` is invalid.

Example task fragment: assume protein token `i = 7` and ligand token `j = 49`.
These illustrative indices are not defaults; replace them with your actual token mapping.

```json
{"constraints": {"contact": {"token_pairs": [[7, 49]]}}}
```
Use `token_pairs` for every pair in this CONTACT object, including protein–protein pairs.

## Combining constraints

SITE and EXCLUSION may use different partner sets. They conflict only when
both their selected residue sets and their partner sets overlap. This is a strict
input rule, even if contact with another, non-excluded partner could satisfy SITE:
`SITE Aᵢ → B` with `EXCLUSION Aᵢ → C` is allowed;
`SITE Aᵢ → B` with `EXCLUSION Aᵢ → {B, C}` is rejected.
CONTACT pairs inside an EXCLUSION scope are also rejected.
Recommended chain counts are **n_source** (by default: = 1) and
**n_partner** (by default: = 1). The recommended CONTACT pair limit is
**K_contact** (by default: = 4); these are not hard inference limits.

## Further reading

See [prediction and weight loading](../README.md#predict-with-mol-1-constraint) for inference commands and checkpoint handling.
Use the [native input format](infer_json_format.md) to construct a full task.
