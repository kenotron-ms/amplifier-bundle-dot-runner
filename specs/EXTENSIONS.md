# Attractor Extensions

Documented divergences and additions relative to the canonical attractor nlspec at
[github.com/strongdm/attractor](https://github.com/strongdm/attractor). The current
canonical snapshot lives at `specs/canonical/attractor-spec-canonical.md`.

**All extensions are backward-compatible with the canonical spec — community `.dot` files
written against the canonical spec should continue to work without modification.**

When in doubt about whether a behavior is spec-conformant, consult the canonical snapshot
before assuming it is a bug.

---

## 1. BareValue Grammar Production

**What:** The value grammar accepts unquoted bare identifiers in addition to quoted strings.
Examples: `shape=box`, `rankdir=LR`, `node_type=llm`. The grammar production is:

```
BareValue ::= [A-Za-z_][A-Za-z0-9_.:-]*
```

**Why:** Graphviz DOT source uses bare identifiers pervasively for built-in shape and
direction attributes. Requiring quotes everywhere would break existing community `.dot`
files. This is an additive clarification of what Graphviz already accepts; it is not a
departure from spec intent.

**Compatibility:** Fully backward-compatible. Quoted values continue to work unchanged.

---

## 2. `default_max_retries` (with Legacy Alias `default_max_retry`)

**What:** The graph-level retry ceiling attribute is `default_max_retries` (plural). The
singular `default_max_retry` is accepted as a legacy alias and maps to the same behavior.
Default value is `0` (no retries unless explicitly configured).

**Why:** The plural form is grammatically clearer ("the number of retries" rather than "the
maximum retry"). The legacy alias ensures any existing `.dot` files using the original
singular name continue to work without modification.

**Compatibility:** Both names are valid. Prefer `default_max_retries` in new pipelines.

---

## 3. `max_retries` Node Attribute Inherits Graph-Level Default

**What:** When a node omits the `max_retries` attribute, it inherits the graph's
`default_max_retries` value rather than defaulting to `0` independently. This allows a
single graph-level setting to establish a retry policy for all nodes simultaneously.

**Why:** Without inheritance, authors must repeat `max_retries=N` on every node that
should participate in a retry policy. The inheritance behavior is the natural complement to
`default_max_retries` existing at all: a graph-level default that nothing inherits would
serve no purpose.

**Compatibility:** Only observable in pipelines that set `default_max_retries` at the
graph level. Pipelines that do not set it see no change (effective retries remain 0).

---

## 4. `goal_gate` Accepts `PARTIAL_SUCCESS`

**What:** A node marked `goal_gate=true` is considered satisfied by either `SUCCESS` or
`PARTIAL_SUCCESS` outcome status. It is NOT satisfied by `FAIL`, `SKIP`, or other
statuses, and the pipeline exits with an unsatisfied-goal error if the node does not
reach at least `PARTIAL_SUCCESS`.

**Why:** Rigid `SUCCESS`-only gate semantics are too coarse for pipelines that implement
best-effort or iterative workflows — for example, a test-generation node that passes most
cases but flags a few as needing human review. Accepting `PARTIAL_SUCCESS` preserves the
gate intent (the node ran and made meaningful progress) while not blocking pipelines that
legitimately reach a partial outcome.

**Compatibility:** Existing `goal_gate=true` nodes that return `SUCCESS` are unaffected.
Nodes that return `PARTIAL_SUCCESS` now satisfy the gate where they previously would have
caused a pipeline failure.

---

## 5. Explicit TRANSFORM Phase in Execution Lifecycle

**What:** The execution lifecycle includes six phases rather than five:

```
PARSE -> TRANSFORM -> VALIDATE -> INITIALIZE -> EXECUTE -> FINALIZE
```

The TRANSFORM phase applies parse-time transforms (stylesheet resolution, variable
expansion, and custom AST transforms) before validation runs.

**Why:** Placing transforms before validation ensures that validation sees the final,
expanded graph — not the template form with unexpanded variables or unresolved stylesheets.
This prevents spurious validation failures on legal pipeline patterns that are valid only
after expansion.

**Compatibility:** Pipeline authors who consume the execution lifecycle events or hook into
the lifecycle will see a new `TRANSFORM` phase event before `VALIDATE`. Pipelines that do
not hook into lifecycle events are unaffected.

---

## 6. Error Semantics: `RETURN Outcome(status=FAIL)` vs `RAISE`

**What:** Handler error paths use `RETURN Outcome(status=FAIL, ...)` rather than raising
exceptions. Unhandled exceptions in handler code are caught and wrapped into a `FAIL`
outcome with the exception message in `notes`.

**Why:** Exception propagation from a handler would terminate the entire pipeline rather
than routing through the graph's conditional edges. Returning a `FAIL` outcome preserves
the pipeline's ability to dispatch to a failure branch (e.g., a `condition="outcome=fail"`
edge to a recovery node or human gate). This is the behavior authors expect: a failed node
should trigger failure-path routing, not crash the pipeline.

**Compatibility:** This is an implementation detail of the engine. Pipeline authors observe
`FAIL` outcomes on handler errors regardless of whether the internal mechanism uses
exceptions or return values. Existing pipelines are unaffected.

---

## 7. `type` vs `node_type` Internal Naming

**What:** The externally visible DOT attribute name for the node handler type is `type`.
The engine may use an internal field named `node_type` to avoid reserved-word conflicts in
Python (where `type` is a built-in). Both names refer to the same concept; the external
behavior is identical.

**Why:** Python's `type` built-in creates naming conflicts in dataclasses and attribute
access. Using `node_type` internally avoids shadowing the built-in. The DOT attribute name
`type` remains canonical and externally visible.

**Compatibility:** Pipeline authors use `type=llm`, `type=parallel`, etc. in DOT source.
The internal renaming is invisible at the DOT level.
