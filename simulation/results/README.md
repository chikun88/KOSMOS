# Offline results provenance

`campaign_heldout.json` and `autotune.json` are historical results from an earlier
controller/model and speed profile. They are not acceptance evidence for the
current sprint configuration or the deployed trajectory tracker. In particular,
`campaign_heldout.json` uses 0.78/0.702 m/s and an earlier clearance model, lacks
progress-abort evidence, and must not be read as a current 1200/1200 pass.

The 2026-10-07 audit records source fingerprints and explicit model limitations:

- [Before fixes](../../docs/SYSTEM_AUDIT_BASELINE_20261007.json): 39/49 arrivals, 8 model contacts, 1 timeout, 1 progress abort.
- [Final configuration/model](../../docs/SYSTEM_AUDIT_MODEL_20261007.json): 42/49 arrivals, 4 model contacts, 2 timeouts, 1 progress abort.
- [Paired RL gate](../../docs/SYSTEM_AUDIT_RL_20261007.json): 89/98 arrivals, 5 model contacts, 1 timeout, 3 progress aborts; the empty residual matches baseline exactly.

The first two campaigns use seed 20261007 and fail the offline acceptance gate.
The RL check uses seed 20260808: paired non-regression passes, but the combined
promotion gate fails on model contacts. These reports use an MPPI-style
lightweight surrogate, not production tracker replay. None observes hardware
collisions or establishes physical acceptance.
