# MemorySplit Architecture Canvas Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a browser-openable Cursor Canvas that presents the MemorySplit evidence, frozen experimental decision, scale assessment, and approved future hybrid architecture.

**Architecture:** One self-contained `.canvas.tsx` file embeds reviewed facts and citations. A compact tab bar switches among Decision, Evidence, Scale, and Future Architecture views; IDE actions open the frozen preregistration and architecture spec.

**Tech Stack:** TypeScript/JSX, `cursor/canvas` built-ins only.

## Global Constraints

- Write exactly one file under the managed Canvas directory.
- Import only from `cursor/canvas`.
- Embed all data inline; no `fetch()` or network calls.
- Use host-theme tokens and built-in components.
- No gradients, box shadows, emojis, decorative borders, or giant text.
- Do not render empty states or invented filler content.
- Every quantitative claim identifies its source.
- Measured evidence and inference are visually distinct.
- The frozen preregistration, not the earlier 90% branch, controls the current battery.
- The Canvas is outside the Git repository and is not committed.

---

### Task 1: Create the complete Canvas

**Files:**
- Create: `/Users/stephenzhang/.cursor/projects/Users-stephenzhang-Documents/canvases/memorysplit-architecture-review.canvas.tsx`

**Interfaces:**
- Consumes: frozen preregistration and approved architecture decision.
- Produces: interactive standalone architecture review.

- [ ] **Step 1: Write the complete Canvas**

Create the file with:

```tsx
import {
  Button,
  Callout,
  Card,
  CardBody,
  CardHeader,
  Divider,
  Grid,
  H1,
  H2,
  H3,
  Link,
  Pill,
  Row,
  Spacer,
  Stack,
  Stat,
  Table,
  Text,
  useCanvasAction,
  useCanvasState,
  useHostTheme,
} from "cursor/canvas";

type Tab = "decision" | "evidence" | "scale" | "architecture";

const SOURCES = {
  lmlm: "https://arxiv.org/abs/2505.15962",
  colmlm: "https://arxiv.org/abs/2607.07707",
  engram: "https://arxiv.org/abs/2601.07372",
  retrievalScale: "https://arxiv.org/abs/2604.00715",
  sparsity: "https://arxiv.org/abs/2508.18672",
};

const TABS: Array<{ id: Tab; label: string }> = [
  { id: "decision", label: "Decision" },
  { id: "evidence", label: "Evidence" },
  { id: "scale", label: "Scale" },
  { id: "architecture", label: "Future architecture" },
];

function SectionCaption({ children }: { children: string }) {
  return (
    <Text size="small" tone="tertiary">
      {children}
    </Text>
  );
}

function DecisionView() {
  const dispatch = useCanvasAction();
  return (
    <Stack gap={18}>
      <Callout tone="success" title="Current program">
        Finish the exact-key, loss-masked 160M sweep and 1B confirmation.
        Fact-use QA is frozen as H1; iGSM and deduction are emergence-watch
        secondaries. Add diagnostics, not a new architecture arm.
      </Callout>

      <Grid columns={4} gap={16}>
        <Stat value="3" label="Gate-A pilot rounds" />
        <Stat value="160M" label="Dose-response scale" />
        <Stat value="1B" label="Confirmation scale" />
        <Stat value="0" label="New training arms now" />
      </Grid>

      <Grid columns="1fr 1fr" gap={16}>
        <Card>
          <CardHeader trailing={<Pill size="sm" active>Claim-bearing</Pill>}>
            Scientific architecture
          </CardHeader>
          <CardBody>
            <Stack gap={8}>
              <Text weight="semibold">Exact-key loss-masked organizer</Text>
              <Text tone="secondary">
                Changes the fewest causal variables: matched decoder, exact
                retrieval, direct value-target masking, and store-ON/OFF
                mediation.
              </Text>
              <Text size="small" tone="tertiary">
                Claim boundary: no direct fact-target loss and zero
                probe-measured fact bits—not provably zero information.
              </Text>
            </Stack>
          </CardBody>
        </Card>

        <Card>
          <CardHeader trailing={<Pill size="sm">Deferred</Pill>}>
            Production follow-up
          </CardHeader>
          <CardBody>
            <Stack gap={8}>
              <Text weight="semibold">Selective hybrid memory hierarchy</Text>
              <Text tone="secondary">
                Continuous queries for broad span retrieval plus a canonical
                exact store for authoritative and editable facts.
              </Text>
              <Text size="small" tone="tertiary">
                A structured 7B pilot follows only after the current battery
                and a natural/noisy-store replication.
              </Text>
            </Stack>
          </CardBody>
        </Card>
      </Grid>

      <H2>Frozen decision path</H2>
      <Table
        headers={["Observed outcome", "Interpretation", "Next action"]}
        rows={[
          [
            "Positive at 160M and 1B",
            "Externalization helps in tested regimes",
            "Natural/noisy replication, then hybrid 7B pilot",
          ],
          [
            "160M positive, 1B null",
            "Scale confounded with 20 vs 10 tokens/parameter",
            "Compute-optimal 1B follow-up before a scale claim",
          ],
          [
            "Defensible null",
            "Control and fact-use benefits, no measured general reasoning gain",
            "Stop architectural expansion for reasoning ROI",
          ],
          [
            "Fact-use gain only",
            "Useful external-memory/tool architecture",
            "Do not claim freed knowledge-free reasoning",
          ],
        ]}
        rowTone={["success", "warning", "neutral", "info"]}
        striped
      />

      <Row gap={8} wrap>
        <Button
          variant="primary"
          onClick={() =>
            dispatch({
              type: "openFile",
              path: "MemorySplit/docs/superpowers/specs/2026-07-20-preregistration.md",
            })
          }
        >
          Open frozen preregistration
        </Button>
        <Button
          variant="secondary"
          onClick={() =>
            dispatch({
              type: "openFile",
              path: "MemorySplit/docs/superpowers/specs/2026-07-20-memory-architecture-scale-decision-design.md",
            })
          }
        >
          Open architecture decision
        </Button>
      </Row>
    </Stack>
  );
}

function EvidenceView() {
  return (
    <Stack gap={18}>
      <H2>Evidence hierarchy</H2>
      <SectionCaption>
        Measured results are separated from extrapolation. Benchmark gains do
        not automatically identify the causal mechanism.
      </SectionCaption>

      <Table
        headers={["Method", "What is stored", "Strongest evidence", "Fit to H1"]}
        rows={[
          [
            <Link href={SOURCES.lmlm}>LMLM / MemorySplit</Link>,
            "Editable nonparametric values; direct value targets masked",
            "Store-off factual collapse at 176M/382M; MemorySplit mechanism replicated",
            "Direct causal fit; reasoning gain still unmeasured",
          ],
          [
            <Link href={SOURCES.colmlm}>Co-LMLM</Link>,
            "Text spans behind continuous vector keys",
            "Large factuality gains at 135M/360M; 90B-token corpus; 2.2B-entry index",
            "Strong follow-up; InfoNCE/index add causal variables",
          ],
          [
            <Link href={SOURCES.engram}>Engram</Link>,
            "Gradient-trained hashed n-gram embeddings",
            "Iso-parameter/FLOP 27B gains: BBH +5.0, ARC-C +3.7, MATH +2.4",
            "Adjacent specialization claim, not facts outside weights",
          ],
          [
            "RETRO / ordinary RAG",
            "External passages; fact targets still trained",
            "Retrieval scales to large models and datastores",
            "Does not remove pretraining memorization pressure",
          ],
        ]}
        rowTone={["success", "info", "warning", "neutral"]}
        striped
      />

      <Grid columns="1fr 1fr 1fr" gap={16}>
        <Card>
          <CardHeader>What MemorySplit has measured</CardHeader>
          <CardBody>
            <Text>
              Store dependence, fresh-entity lookup generalization, zero
              probe-measured split fact bits, and strong fact-use QA.
            </Text>
          </CardBody>
        </Card>
        <Card>
          <CardHeader>What remains unmeasured</CardHeader>
          <CardBody>
            <Text>
              A replicated knowledge-free reasoning gain and a clean trend
              from 160M to a compute-matched 1B point.
            </Text>
          </CardBody>
        </Card>
        <Card>
          <CardHeader>Strongest negative bound</CardHeader>
          <CardBody>
            <Text>
              Retrieval often helps factual access without improving
              self-contained reasoning; freed capacity is not automatically
              reallocated usefully.
            </Text>
          </CardBody>
        </Card>
      </Grid>

      <Callout tone="warning" title="Inference, not measurement">
        A hybrid 7B design is plausible, but Co-LMLM stops at 360M and Engram
        tests trainable memory. Neither result proves that MemorySplit's current
        intervention improves frontier-model reasoning.
      </Callout>

      <Row gap={14} wrap>
        <Link href={SOURCES.retrievalScale}>Retrieval/pretraining scaling</Link>
        <Link href={SOURCES.sparsity}>Optimal MoE sparsity for reasoning</Link>
      </Row>
    </Stack>
  );
}

function ScaleView() {
  return (
    <Stack gap={18}>
      <H2>Where memory belongs in the optimization stack</H2>
      <Table
        headers={["Regime", "Memory opportunity", "Higher-priority work", "Decision"]}
        rows={[
          [
            "1–3B",
            "Highest chance of effective-capacity pressure",
            "Learnable endpoint, data exposure, paired seeds",
            "Run the protected battery",
          ],
          [
            "7B–50B",
            "Selective long-tail, private, volatile, attributable facts",
            "Data mix, active compute, retrieval/serving measurement",
            "Structured pilot only after replication",
          ],
          [
            "Frontier MoE",
            "Conditional memory as one sparse-capacity tier",
            "Mixture/repetition, active FLOPs, routing, interconnect, stability, serving",
            "No frontier MemorySplit run now",
          ],
        ]}
        rowTone={["success", "info", "warning"]}
        striped
      />

      <Grid columns="1fr 1fr" gap={16}>
        <Card>
          <CardHeader>Why the 1B point is qualified</CardHeader>
          <CardBody>
            <Text>
              The 160M sweep uses about 20 tokens/parameter; the 1B
              confirmation uses about 10. A disagreement is scale plus
              training-regime evidence, not a pure scale law.
            </Text>
          </CardBody>
        </Card>
        <Card>
          <CardHeader>When external memory is first-order</CardHeader>
          <CardBody>
            <Text>
              Freshness, deletion, provenance, private knowledge, and
              long-tail factual coverage can justify external memory even
              without a general reasoning gain.
            </Text>
          </CardBody>
        </Card>
      </Grid>

      <H3>Priority order at frontier scale</H3>
      <Table
        headers={["Priority", "Lever"]}
        rows={[
          ["1", "Data quality, mixture, repetition, curriculum"],
          ["2", "Active FLOPs, model shape, tokens per parameter"],
          ["3", "MoE routing, expert placement, interconnect"],
          ["4", "Optimizer and numerical stability"],
          ["5", "Attention, KV cache, batching, serving"],
          ["6", "Post-training and adaptive test-time compute"],
          ["7", "Conditional/external memory where product needs justify it"],
        ]}
        columnAlign={["right", "left"]}
      />
    </Stack>
  );
}

function ArchitectureView() {
  const theme = useHostTheme();
  const nodeStyle = {
    padding: 14,
    border: `1px solid ${theme.stroke.secondary}`,
    borderRadius: 8,
    background: theme.fill.tertiary,
  };
  return (
    <Stack gap={18}>
      <H2>Approved future hybrid memory hierarchy</H2>
      <SectionCaption>
        Inference flow; this is a post-battery design, not a current training arm.
      </SectionCaption>

      <Grid columns="1fr auto 1fr auto 1fr" gap={10} align="center">
        <div style={nodeStyle}>
          <Text weight="semibold">Backbone</Text>
          <Text size="small" tone="secondary">
            Schemas, procedures, language, common stable knowledge
          </Text>
        </div>
        <Text tone="tertiary">→</Text>
        <div style={nodeStyle}>
          <Text weight="semibold">Continuous router</Text>
          <Text size="small" tone="secondary">
            Retrieve, abstain, batch, and record trace
          </Text>
        </div>
        <Text tone="tertiary">→</Text>
        <div style={nodeStyle}>
          <Text weight="semibold">Evidence returned</Text>
          <Text size="small" tone="secondary">
            Versioned value, provenance, confidence
          </Text>
        </div>
      </Grid>

      <Grid columns="1fr 1fr 1fr" gap={16}>
        <Card>
          <CardHeader>Canonical exact store</CardHeader>
          <CardBody>
            <Text>
              Typed entities/relations, aliases, temporal qualifiers,
              set-valued facts, versioned edits. Authoritative path.
            </Text>
          </CardBody>
        </Card>
        <Card>
          <CardHeader>Dense span index</CardHeader>
          <CardBody>
            <Text>
              Free-form source-faithful evidence for knowledge that does not
              fit a fixed schema. Coverage path.
            </Text>
          </CardBody>
        </Card>
        <Card>
          <CardHeader>Conditional hot cache</CardHeader>
          <CardBody>
            <Text>
              Optional Engram-like tier for stable local patterns. An
              efficiency component, not nonparametric memory.
            </Text>
          </CardBody>
        </Card>
      </Grid>

      <Divider />
      <H3>Required controls before a 7B pilot</H3>
      <Table
        headers={["Control", "Why it is required"]}
        rows={[
          ["Exact versus continuous query", "Separate interface quality from externalization"],
          ["Mask-only arm", "Separate gradient removal from usable memory"],
          ["Natural call timing", "Forced lookups hide under-retrieval"],
          ["Full latency accounting", "Query formation excludes ANN search and returned-span prefill"],
          ["Versioned index projection", "Post-training can invalidate hidden-state keys"],
          ["Natural/noisy fact store", "Synthetic exact keys overstate retrieval cleanliness"],
        ]}
        striped
      />
    </Stack>
  );
}

export default function MemorySplitArchitectureReview() {
  const [tab, setTab] = useCanvasState<Tab>("memorysplit-tab", "decision");
  const dispatch = useCanvasAction();

  return (
    <Stack gap={20} style={{ maxWidth: 1080 }}>
      <Stack gap={5}>
        <Row align="center" wrap>
          <H1>MemorySplit Architecture Review</H1>
          <Spacer />
          <Button
            variant="ghost"
            onClick={() =>
              dispatch({
                type: "newComposerChat",
                userPrompt:
                  "Review the MemorySplit architecture decision and identify any evidence that should change the current go/no-go boundary.",
              })
            }
          >
            Reassess in chat
          </Button>
        </Row>
        <Text tone="secondary">
          Exact organizer for causal science; selective hybrid memory for a
          conditional production follow-up.
        </Text>
        <Text size="small" tone="tertiary">
          Sources: frozen preregistration and architecture decision, Jul 20,
          2026; primary papers linked in Evidence.
        </Text>
      </Stack>

      <Row gap={8} wrap>
        {TABS.map((item) => (
          <Pill
            key={item.id}
            active={tab === item.id}
            onClick={() => setTab(item.id)}
          >
            {item.label}
          </Pill>
        ))}
      </Row>

      {tab === "decision" && <DecisionView />}
      {tab === "evidence" && <EvidenceView />}
      {tab === "scale" && <ScaleView />}
      {tab === "architecture" && <ArchitectureView />}
    </Stack>
  );
}
```

- [ ] **Step 2: Confirm the Canvas TypeScript check**

Expected tool result:

```text
Canvas TypeScript check: no errors
```

If an export or prop is invalid, read the matching `.d.ts` file under
`~/.cursor/skills-cursor/canvas/sdk/`, correct the code, and rerun the check.

### Task 2: Perform the visual and content self-check

**Files:**
- Verify: `/Users/stephenzhang/.cursor/projects/Users-stephenzhang-Documents/canvases/memorysplit-architecture-review.canvas.tsx`

**Interfaces:**
- Consumes: Task 1.
- Produces: approved standalone analytical artifact.

- [ ] **Step 1: Open every tab**

Verify:

```text
- Decision contains the frozen fact-use primary, not the superseded 90% branch.
- Evidence distinguishes measured evidence from inference.
- Scale distinguishes 1–3B, 7B–50B, and frontier MoE.
- Future architecture shows exact, dense-span, and optional conditional tiers.
- No tab is empty.
```

- [ ] **Step 2: Run the design self-check**

Confirm:

```text
- one dominant page title and clear section hierarchy;
- open sections mixed with bounded cards;
- no wall of identical cards;
- no gradients, emojis, box shadows, rainbow colors, or decorative borders;
- no text above 24px;
- tables have self-explanatory headers;
- quantitative claims identify source context;
- light and dark host themes remain readable.
```

- [ ] **Step 3: Test IDE actions**

Click:

```text
- Open frozen preregistration
- Open architecture decision
- Reassess in chat
```

Expected: both files open in the IDE; the chat action opens a new composer with
the Canvas attached.

- [ ] **Step 4: Re-run the TypeScript check after any corrections**

Expected:

```text
Canvas TypeScript check: no errors
```

- [ ] **Step 5: Record the artifact link for handoff**

Use:

```markdown
[MemorySplit architecture review](/Users/stephenzhang/.cursor/projects/Users-stephenzhang-Documents/canvases/memorysplit-architecture-review.canvas.tsx)
```

Do not `git add` or commit the Canvas; it lives in Cursor's managed project
directory, outside `MemorySplit`.

## Self-Review

- Spec coverage: causal/production split, evidence hierarchy, current battery,
  scale regimes, future data flow, controls, and primary sources are present.
- Content scan: every section contains reviewed, nonempty data.
- API consistency: every imported component appears in `cursor/canvas` public
  declarations and only supported props are used.
