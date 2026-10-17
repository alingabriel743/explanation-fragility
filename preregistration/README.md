# Pre-registration

| File | Content |
|---|---|
| `PREREGISTRATION_cies_variants.md` | Modifications of CIES (V1 to V4), Amendment A1 (bounded aggregation, adopted as CIES) and Amendment A2 (decision-consequence bound) |
| `PREREGISTRATION_A3_monitoring.md` | The monitoring task with a fixed review budget, evaluated on the fresh seeds 5 to 9 |

Each document was hashed before the results it governs were computed. The hashes can be checked with

```sh
shasum -a 256 -c PREREGISTRATION_cies_variants.sha256
head -1 PREREGISTRATION_A3_monitoring.sha256 | shasum -a 256 -c
```

The second hash file also records the local time at which the hash was taken. The hashes document the content of the registrations. Their timing rests on the preserved file timestamps and on the order in which the data were generated.
