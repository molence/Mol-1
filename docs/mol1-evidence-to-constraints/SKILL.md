---
name: mol1-evidence-to-constraints
description: Convert experimental evidence from papers or records into proposed Mol-1 SITE, EXCLUSION, or CONTACT inputs. Preserve provenance, map residue and token endpoints, and validate native JSON and constraint compatibility.
---

# Evidence → Mol-1 constraints

Example request: “Convert this paper's experimental evidence into constraints for my input.json, and show the conversion and validation results.”

Produce a traceable chain: **source context → observation and original labels → proposed meaning → input numbering → JSON → verification**. The user judges scientific appropriateness; the agent ensures the chosen meaning is encoded correctly. Do not add evidence-sufficiency gating or require scientific approval merely to generate a proposal.

## Workflow

### 1. Establish the input and evidence

Use evidence supplied by the user or encountered during research. Reuse the target native task, its exact sequences, chain instances, and existing constraints. Identify the target Mol-1 version. Do not change sequences or chain IDs merely to simplify mapping.

For each evidence item, record source type, title or record ID, URL/file, exact locator, relevant context, observation, original residue labels, and interacting entities. Distinguish paper body, abstract, supplement, database annotation, and user record. Respect requested source restrictions. Separate experimental evidence from numbering references, reported results from computed quantities, and both from your interpretation. Mark missing provenance rather than inventing it.

If the input is absent, extract observations and propose types, then request the sequences or task needed for executable conversion. Create a reference input only when the user requests an example or authorizes that choice; declare its sequences and constructs.

### 2. Make the semantic connection

Follow the user's selected interpretation; otherwise make a concrete, labeled proposal using the [native constraint semantics](../constraint_semantics.md). A single observation may motivate one or several compatible constraint types.

| Observation | Candidate encoding | Choice to explain |
| --- | --- | --- |
| Mutational effects, chemical shift perturbation, residue labeling | SITE | Selected residues and partner scope |
| HDX protection, footprinting, peptide mapping | SITE | Which positions are selected; whether a whole region becomes per-residue conditions |
| Crosslink, NOE, FRET/DEER, pairwise evidence | CONTACT | Exact endpoints; original measurement versus the fixed native contact relation |
| Weak/negative response, nonbinding region, undesired interface | EXCLUSION | Which residues and partners are assigned geometric separation |

These are encoding routes, not assay-to-geometry equivalences. Preserve differences in distance, resolution, or meaning in the conversion record. Do not mark every unreported residue as EXCLUSION. Competition or epitope binning needs an identified region before residue constraints can be generated; a binder relationship alone supplies no residue endpoints.

### 3. Map, combine, and serialize

Apply the mapping rules and native contract below, using the target's [input and numbering conventions](../constraint_semantics.md#input-and-numbering). Map original labels to the exact input's **1-based label_seq_id**. Use actual preprocessing to obtain **0-based global token indices** and verify projected endpoint identities.

Combine proposals with existing constraints only when their meanings and [native combination rules](../constraint_semantics.md#combining-constraints) are compatible. Apply an explicit user priority to conflicts and record any omission; otherwise provide separately named alternatives. Separate tasks are alternative evaluations, not a way to enforce independent groups jointly. Never silently drop constraints or broaden partner scope.

### 4. Verify and deliver

Use the target implementation's parser, preprocessing, and constraint projectors. The inspected snapshot supports:

`inputs.read_json` → `inputs.prepare_inputs` → `InferenceDataset.process_one` → `features.project_sites` → `features.exclusion_project` → `features.contact_project` → `runtime.require_finite`.

Check current signatures and configuration. Pass the prepared per-task descriptors to the projectors, carrying the returned features forward so combined conflicts are checked. Disable MSA and templates for this validation; use real feature/atom objects. No prediction or checkpoint loading is required. CONTACT/EXCLUSION validation requires their corresponding configuration flags; enabling these flags for a technical check does not establish checkpoint compatibility.

Check each task's original-to-input correspondence independently of parser acceptance. Verify requested versus projected residues/pairs, partner scope, token membership, supported fields, ranges, conflicts, and finite features. A wrong but in-range index can pass native validation. When providing both CONTACT input forms, verify that they project to the same undirected token-pair set.

Deliver only the requested artifacts:

- **Native input:** `input.constraints.json`, a complete task list preserving unrelated fields. Label a partial JSON object `constraints.fragment.json`; it is not a runnable input.
- **Conversion record:** source/context → observation + original labels → interpretation → target chain/label_seq_id/token scope → JSON entry → check result. Include residue identities and mapping basis.
- **Validation status:** distinguish *technically validated proposal*, *awaiting native validation*, and *unresolved mapping or conflict*. Record the input identity, target version, configuration, and actual checks performed. Disclose any unverified checkpoint compatibility separately.

Unresolved endpoints or conflicts prevent a ready designation for affected inputs; unaffected items may be delivered separately with omissions identified. If the runtime is unavailable, do not fabricate tokens or claim native validation. Preserve a residue-based draft when possible. Keep provenance and measurements outside `constraints`. Technical validation does not prove scientific appropriateness, geometric satisfaction, or prediction quality.

## Residue and token mapping

### Residue endpoints

`[chain_id, position]` refers to the complete sequence actually submitted for that chain, including any truncation or tags. Source author numbering, mature-protein numbering, insertion codes, and even a source mmCIF's label numbering must be mapped to this input.

1. Establish source entity, numbering namespace, chain/copy, original labels, and a sequence or explicit numbering reference. Preserve positions with missing coordinates; do not count visible atoms as sequence positions.
2. Map to the exact input sequence by verified alignment or explicit conversion. Check identities, cleavage, truncation, substitutions, and insertions/deletions. An offset requires a documented basis and sequence checks, not one matching residue.
3. Record original label, reference position, input chain, label_seq_id, source/target identities, and mapping basis for each exported residue. Resolve ambiguous alignments, absent positions, and ambiguous chain copies; do not propagate evidence to all copies automatically.

The numbering reference need not reproduce the exact assayed construct. Declare the distinction and known substitutions. Scientific transfer across constructs or conditions is the user's decision; positional correspondence remains a technical requirement.

### Token endpoints

Use the actual featurized token axis for the current task and preprocessing configuration. Retain the input identity and token-to-chain/residue/atom mapping. Regenerate it after sequence, entity order, copy, modification, bond, or preprocessing changes.

Tokens are not chain-local positions or general atom indices. Do not derive them solely by subtracting one, assuming chain concatenation, or assuming one token per ligand. Inspect all atoms in an endpoint token: a multi-atom token does not encode an atom-specific constraint. Residue-pair CONTACT endpoints must each resolve to exactly one protein token.

If coordinates supply evidence or identities, retain model/assembly, chain namespaces, ligand instances, and atom selection. Label computed distances and relevant missing atoms or alternate conformers. A minimum over observed atoms is not necessarily a complete-set minimum. Coordinates used as evidence do not automatically become templates.

## Native input contract

Snapshot: mol-1-dev inspected on 2026-11-04. Check the target repository's [constraint semantics](../constraint_semantics.md), [input JSON format](../infer_json_format.md), and constraint validation implementation for version differences. If they disagree, resolve the discrepancy before claiming compatibility.

A complete input follows the [native JSON task format](../infer_json_format.md): a list of task objects, each with its own optional `constraints` object. Use only supported native fields; do not mix legacy constraint paths into the generated task. Do not add measurement, confidence, weight, or distance fields to `constraints`.

### SITE and EXCLUSION

- Each task accepts at most one object of each type: `source_chains`, `partner_chains`, and `residue_ids`.
- Active chain lists are nonempty and unique; source and partner sets are disjoint. Selected residues belong to source chains and resolve to protein tokens. Partner scope includes **every token** in the declared partner chains and can include non-proteins.
- SITE asks **each selected residue** to contact **any member** of the combined partner set. It does not mean that any one selected residue suffices, or that every partner chain must be contacted.
- EXCLUSION asks each selected residue to remain separated from **every member** of that set. The minimum heavy-atom distance to the full partner scope must be strictly greater than the configured exclusion threshold (snapshot default: 6 Å). Equality fails the geometric condition; this is not a prediction rejection filter.
- All residues in an object share its partner set. Combining `A:3 → B` and `A:5 → C` as `{A:3,A:5} → {B,C}` changes the meaning and is not an equivalent encoding.

### CONTACT

- Use either `pairs` (protein `[chain_id, positive_integer]` endpoints) or `token_pairs` (nonnegative integer indices), never both in one object. Non-protein endpoints require `token_pairs`; convert all pairs in that object to this form.
- Endpoints must differ. Pairs are undirected and duplicates are deduplicated. Same-chain and adjacent-residue pairs are allowed. Unlisted pairs are unknown, not negative.
- CONTACT asks for minimum heavy-atom distance ≤ the checkpoint's contact threshold (snapshot default: 4.5 Å). The input cannot specify an arbitrary experimental cutoff.

### Compatibility and inactive conditions

- SITE and EXCLUSION conflict when **both** their selected residue sets and partner sets overlap. This is a strict input rule even if a non-excluded partner could satisfy SITE.
- CONTACT conflicts with EXCLUSION if either endpoint is an excluded source residue/token and the other lies in that exclusion's partner scope. Check both orientations after token projection.
- EXCLUSION is not the logical complement of CONTACT: the default cutoffs leave a gap between 4.5 and 6 Å.
- Omitted, `null`, or `{}` constraints are inactive. Empty residue/pair lists also provide no condition. A nonempty SITE/EXCLUSION object lacking `residue_ids` is invalid. Unsupported fields remain errors; do not rely on inactive conditions to check chain existence.
- Active chains, residues, and tokens must exist after preprocessing. Multiple source/partner chains and more than four CONTACT pairs are supported; training recommendations are not hard input limits. Do not truncate valid input to meet recommendations.
- The intended deployment must support the selected types. Input compatibility checks do not establish that any particular checkpoint can use them, and constraints do not guarantee prediction satisfaction.

## Example: from an observation to validated constraints

This self-contained example uses invented observations and arbitrary short sequences to demonstrate the workflow. It does not reproduce a paper or describe a biological finding. All example inputs are shown below.

### 1. Read the context and identify the original labels

The user supplies a record named `example-record-01`, with this context:

> Alpha residues 101–110 have sequence ACDEKFGHIL; Beta residues 201–208 have sequence MGKSTVRA. Both fragments have no gaps or substitutions. With Beta present, Alpha D103 and K105 show chemical-shift changes relative to Alpha alone, while Alpha L110 shows weak response. A crosslink is assigned between Alpha K105 and Beta K203.

Record the source as a user-supplied record, with the sequence statement and observations above as its locators. The modeling task uses the Alpha fragment as chain A and the Beta fragment as chain B.

### 2. State the proposed interpretation

| Observation | Proposed constraint | Interpretation to disclose |
| --- | --- | --- |
| Alpha D103 and K105 show chemical-shift changes | SITE toward chain B | Select these residues as contacting the partner |
| Alpha L110 shows weak response | EXCLUSION toward chain B | Represent weak response as geometric separation |
| Alpha K105–Beta K203 crosslink | CONTACT between the two residues | Represent the crosslink as the native fixed contact relation |

The latter two interpretations do not turn the observations into measured distances. Preserve this distinction in the conversion record; scientific acceptance remains with the user.

### 3. Convert the numbering and check identities

Chain A starts at source position 101: subtract 100. Chain B starts at 201: subtract 200. The supplied contiguous sequences justify these offsets.

| Original label | Input label_seq_id | Identity check against input |
| --- | --- | --- |
| Alpha D103 | A3 | AC**D**EKFGHIL → D |
| Alpha K105 | A5 | ACDE**K**FGHIL → K |
| Alpha L110 | A10 | ACDEKFGHI**L** → L |
| Beta K203 | B3 | MG**K**STVRA → K |

### 4. Construct the complete native input

Preserve the declared sequences and chain IDs, and add the three compatible constraints:

```json
[
  {
    "name": "example_alpha_beta",
    "sequences": [
      {"proteinChain": {"sequence": "ACDEKFGHIL", "count": 1, "id": ["A"]}},
      {"proteinChain": {"sequence": "MGKSTVRA", "count": 1, "id": ["B"]}}
    ],
    "constraints": {
      "site": {
        "source_chains": ["A"], "partner_chains": ["B"],
        "residue_ids": [["A", 3], ["A", 5]]
      },
      "exclusion": {
        "source_chains": ["A"], "partner_chains": ["B"],
        "residue_ids": [["A", 10]]
      },
      "contact": {"pairs": [[["A", 5], ["B", 3]]]}
    }
  }
]
```

### 5. Verify the input and inspect actual tokens

The native preprocessing and projectors were checked for these sequences and constraints on the inspected snapshot. They produced 18 tokens and the following endpoint mapping:

| Input residue | Actual token index (0-based) | Use |
| --- | --- | --- |
| A3 | 2 | SITE |
| A5 | 4 | SITE and CONTACT |
| A10 | 9 | EXCLUSION |
| B3 | 12 | CONTACT partner |

The projected partner scope for SITE and EXCLUSION contains all eight B tokens. Two SITE residues, one EXCLUSION residue, and one CONTACT pair were retained. SITE and EXCLUSION select disjoint residues; neither CONTACT endpoint is the excluded A10 residue. Native format, projection, conflict, and finite-feature checks passed.

To express the same CONTACT with tokens, replace its entire object with:

```json
{"token_pairs": [[4, 12]]}
```

Do not retain `pairs` alongside it. Both representations were verified to project to the same pair. In a new task, obtain the mapping from its actual preprocessing; do not reuse these token values or infer them from chain lengths.

### 6. Present the result

A concise delivery for this example is: “The proposed input contains two SITE residues, one EXCLUSION residue, and one CONTACT pair. Original labels map to A3, A5, A10, and B3 with matching identities. Native validation passes, and the CONTACT maps to token pair [4,12]. The weak-response and crosslink interpretations remain scientific choices.”

For a new execution, report its own validation status. The example's recorded result does not establish that another input or environment has been validated.
