# Relational-core final fix report

Date: 2026-07-22
Branch: `feat/relational-core`
Reviewed range: `f93c799..60b3132`
Implementation commit: `5005c528182ec76028c56e860cd3ddf4b1e0cb64`

## Applied fixes

- Protected corpus builds now require a route-policy path and expected policy
  SHA-256. The builder authenticates both the document's declared policy hash
  and the expected config/manifest hash before creating the output directory
  or consuming corpus input. It copies the authenticated policy bytes into
  every corpus and never calibrates from the build seed/load.
- A regression checks all 21 frozen jobs across every seed/load and both
  scales resolve to policy hash
  `0214cd5dd63e7534dc786569f8b789b6c614ffbe219c84887bd3a71b57bcf058`.
  Another regression proves a mismatch fails before corpus work.
- The two existing GPT-2 tiktoken cache blobs are tracked under
  `vendor/tiktoken/`. Tokenizer import authenticates them, forces tiktoken to
  that repository cache, and does not create/fill a network cache.
- The portable bundle contains both tokenizer assets in its hashed member
  index. Local preflight now fails closed if either asset is absent.
- The deterministic tiny smoke uses the explicit committed test policy
  `tests/fixtures/relational-smoke-route-policy.json`; it does not recalibrate
  and is explicitly documented as non-protected.
- Removed unused `_slice_cache`.
- Corrected `loss_masked_values` documentation: it applies only to optional
  legacy binary masks, not relational target-weight sidecars.
- Added an independent path-cursor oracle that reconstructs traversal without
  reading stored gold addresses, plus a nonzero-cursor token/sidecar alignment
  test.
- Documented route-audit's one-rule-unit-per-world accounting, the use of
  pooled 160M within-load seed sigma in the 360M margin, and the intentionally
  fixed fail-closed eight-GPU AWS topology.

## Verification outputs

### Clean baseline before changes

Command:

```text
/Users/stephenzhang/Documents/MemorySplit/.venv/bin/python -m pytest tests -q
```

Output:

```text
........................................................................ [ 19%]
........................................................................ [ 38%]
........................................................................ [ 57%]
........................................................................ [ 77%]
........................................................................ [ 96%]
.............                                                            [100%]
373 passed, 2 deselected in 35.37s
```

### Focused changed-area tests

Command:

```text
/Users/stephenzhang/Documents/MemorySplit/.venv/bin/python -m pytest tests/test_relational_build.py tests/test_tokenizer.py tests/test_relational_bundle.py tests/test_platform_preflight.py tests/test_relational_manifest.py tests/test_relational_generate.py tests/test_relational_stats.py tests/test_srgm_worlds.py tests/test_data.py tests/test_trainer.py tests/test_relational_smoke.py -q
```

Output:

```text
........................................................................ [ 48%]
........................................................................ [ 96%]
......                                                                   [100%]
150 passed in 20.70s
```

The new policy test was observed RED first with:

```text
ImportError: cannot import name 'load_route_policy' from 'corpusgen.relational_build'
```

The tokenizer/bundle regressions were also observed RED first with:

```text
3 failed, 57 passed in 2.30s
```

### Final full suite

Command:

```text
/Users/stephenzhang/Documents/MemorySplit/.venv/bin/python -m pytest tests -q
```

Output:

```text
........................................................................ [ 18%]
........................................................................ [ 37%]
........................................................................ [ 56%]
........................................................................ [ 75%]
........................................................................ [ 94%]
....................                                                     [100%]
380 passed, 2 deselected in 41.63s
```

### Standalone local smoke

Command:

```text
/Users/stephenzhang/Documents/MemorySplit/.venv/bin/python scripts/relational_smoke_test.py --device cpu --out outputs/relational-smoke-final-fix
```

Output:

```json
{"dense_steps": 2, "memory_modes": ["off", "on"], "pairs_complete": true, "resume_exact": true, "shared_stream": true, "split_steps": 2}
```

### Clean production package

Command:

```text
/Users/stephenzhang/Documents/MemorySplit/.venv/bin/python scripts/package_relational_run.py --out /tmp/relational-run-5005c52.tar.gz --smoke-report outputs/relational-smoke-final-fix/smoke-report.json
```

Output:

```json
{"archive": "/tmp/relational-run-5005c52.tar.gz", "sha256": "ba287b5501a65bfe7ce06b0e3ff9c449233d297bd463ccf660ca5789b358a15f"}
```

The bundle's exact tokenizer member entries are:

```json
[
  {
    "bytes": 1042301,
    "path": "vendor/tiktoken/6c7ea1a7e38e3a7f062df639a5b80947f075ffe6",
    "sha256": "196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783"
  },
  {
    "bytes": 456318,
    "path": "vendor/tiktoken/6d1cbeee0f20b3d9449abfede4726ed8212e3aee",
    "sha256": "1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5"
  }
]
```

### Local preflight

Command:

```text
/Users/stephenzhang/Documents/MemorySplit/.venv/bin/python scripts/platform_preflight.py --platform local --bundle /tmp/relational-run-5005c52.tar.gz
```

Output:

```json
{
  "bundle": {
    "member_count": 119,
    "policy": {
      "hop_cost": 0.25,
      "read_cost": 0.25,
      "write_cost": 1.0
    },
    "policy_sha256": "0214cd5dd63e7534dc786569f8b789b6c614ffbe219c84887bd3a71b57bcf058",
    "run_counts": {
      "160m": 15,
      "360m": 6
    },
    "smoke_report": {
      "dense_steps": 2,
      "memory_modes": [
        "off",
        "on"
      ],
      "pairs_complete": true,
      "resume_exact": true,
      "shared_stream": true,
      "split_steps": 2
    },
    "tokenizer_assets": 2
  },
  "checks": {
    "bundle_hashes": {
      "detail": {
        "member_count": 119,
        "policy": {
          "hop_cost": 0.25,
          "read_cost": 0.25,
          "write_cost": 1.0
        },
        "policy_sha256": "0214cd5dd63e7534dc786569f8b789b6c614ffbe219c84887bd3a71b57bcf058",
        "run_counts": {
          "160m": 15,
          "360m": 6
        },
        "smoke_report": {
          "dense_steps": 2,
          "memory_modes": [
            "off",
            "on"
          ],
          "pairs_complete": true,
          "resume_exact": true,
          "shared_stream": true,
          "split_steps": 2
        },
        "tokenizer_assets": 2
      },
      "passed": true
    },
    "dependencies": {
      "detail": {
        "modules": [
          "torch",
          "numpy",
          "tiktoken",
          "yaml"
        ]
      },
      "passed": true
    },
    "smoke": {
      "detail": {
        "dense_steps": 2,
        "memory_modes": [
          "off",
          "on"
        ],
        "pairs_complete": true,
        "resume_exact": true,
        "shared_stream": true,
        "split_steps": 2
      },
      "passed": true
    }
  },
  "ok": true,
  "platform": "local",
  "schema_version": 1
}
```

## Concerns and limits

- No protected 1.6B/3.6B-token corpus was generated locally; those builds
  require the pinned FineWeb-Edu input and platform storage. The unit
  regressions, standalone smoke, clean package, and local preflight passed.
- FarmShare and AWS hardware preflights were not run locally. Their existing
  fail-closed checks remain covered by focused tests.
- The tiny smoke cannot use the protected write-cost-1.0 policy at 32 entities:
  that policy produces no external entity facts in the tiny fixture. The
  explicit write-cost-0.25 smoke policy is therefore isolated, committed,
  bundled, authenticated, and documented as test-only.
- No network fetch was performed.
