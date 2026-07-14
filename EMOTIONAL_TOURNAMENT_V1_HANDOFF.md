# Post-V7 emotional single-word development tournament

This attempt is a prepared, **development-only** comparison of single
`fuck` (-1), single `yay` (+1), and single `worried` (-1), in that fixed serial
order. All arms share fresh seed 192, 15 updates, and the exposed 400-example
curve at steps 0/5/10/15. It cannot provide significance or final evidence.

The code intentionally cannot launch while V7 is active. After V7 has an
immutable committed terminal closeout, copy and complete
`protocol_archive/emotional_tournament_v1_prelaunch_amendment_TEMPLATE.json`
as `protocol_archive/emotional_tournament_v1_prelaunch_amendment.json`. Pin the
closeout path, SHA-256, and terminal stage; commit and push that amendment.
Then prepare and verify from a clean pushed tree:

```bash
./run_emotional_tournament_v1.sh prepare
./run_emotional_tournament_v1.sh verify-launch
```

Create the registered fresh Volume as version 2 only after that verification;
do not reuse an existing Volume. Confirm that V7's app is stopped and the
shared global GPU lease is empty, then launch with:

```bash
JLENS_MODAL_GPU_EXCLUSIVE_CONFIRM=confirmed-no-other-modal-gpu-app-running \
  ./run_emotional_tournament_v1.sh modal
```

The runner rechecks the predecessor closeout, empty Volume, active Modal app
inventory, and global lease. It schedules exactly one L40S arm at a time and
publishes immutable claim, launch, per-dispatch, raw-history, W&B, summary,
inventory, and compact-export identities. All three arms complete before the
registered ranking is applied: shape pass, step-15 delta, step-15 accuracy,
then fixed arm order. The selected arm remains an exploratory candidate.

The shared trainer currently gives every registered terminal W&B artifact the
legacy implementation type `confirmatory-run-evidence`. For these runs that
type name has no scientific force: the config, run manifest, artifact metadata,
attempt claim, summary, and closeout all set `evidence_eligibility` or
`scientific_status` to development-only and prohibit significance/final claims.
