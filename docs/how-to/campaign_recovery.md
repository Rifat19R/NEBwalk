# Campaign recovery and audit trail

Campaign state is an atomically replaced JSON document guarded by an exclusive
lock. Every durable transition also appends an event with the stage, iteration,
timestamp, and artifact hashes. Failures record their type, message, and
traceback before releasing the lock.

```bash
NEBwalk campaign status campaign-output
NEBwalk campaign resume campaign.json
```

Resume verifies the state and re-enters the first incomplete stage. Completed
QE labels, dataset versions, model manifests, and registry entries are reused
only when their required metadata and checksums remain valid. Do not manually
edit state or manifests. Restore corrupt artifacts from a known backup or start
a new campaign identifier.

Only one process may own a campaign directory. A lock conflict is a hard stop;
confirm that the original process is gone before removing a stale lock. Keep
campaign directories on storage with reliable atomic rename and exclusive-file
creation semantics.

`--force` records an intentional resume request but does not delete, invalidate,
or repeat completed durable stages; terminal campaigns remain terminal. It does
not make corrupt or incompatible artifacts trustworthy. Preserve the full campaign directory for
auditing, including raw calculator outputs, events, dataset manifests, model
checksums, stopping decisions, and final-validation records.

Campaign JSON is trusted configuration, not executable code. Endpoint paths may
refer outside the campaign directory, but every generated artifact recorded by
the state store must resolve inside `campaign_dir`. Use reviewed local JSON;
NEBwalk does not fetch or execute code named by campaign configuration.
