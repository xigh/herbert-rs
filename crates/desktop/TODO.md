# Desktop TODO

## Vision GPU Progress — Approach B (preferred)

Replace `waitUntilCompleted` with `addCompletedHandler` + `Condvar` so that
`MTLSharedEvent` listeners can fire during GPU execution (single CB, no split).

Current state: Approach C (polling `signaledValue`) is implemented.
Approach B is cleaner but needs investigation on dispatch queue scheduling.

See logs in `/tmp/herbert-vision.log` — all 24 SharedEvent notifications
currently fire AFTER `waitUntilCompleted` returns because the blocked thread
prevents the listener's dispatch queue from processing callbacks on M1.
