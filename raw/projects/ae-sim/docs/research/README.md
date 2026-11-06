# Research notes

Working notes behind the project, in reading order.

| # | note | one line |
|---|---|---|
| 01 | [Replication targets](01-replication-targets.md) | Which published AE experiments report numbers we can check against, and what code each needs |
| 02 | [Rintoul: heterodyning tank, in vivo, recording](02-rintoul-heterodyning-and-in-vivo.md) | The best-specified open experiments; the 20–40× simulation gap; the replication plan |
| 03 | [Rodent head models](03-rodent-head-models.md) | SIGMA v2 rat and Digimouse whole-head datasets, `create_mesh`, what they unlock |
| 04 | [The coupling constant K](04-coupling-constant.md) | Literature values span 0.02–2 % / MPa; ours is 0.1; why it must be a parameter |
| 05 | [Prototype findings](05-prototype-findings.md) | Physics lessons, baseline numbers, verification record, hosts |
| 06 | [Roadmap](06-roadmap.md) | Code changes, replications, model building, in dependency order |
| 07 | [Data and tutorials](07-data-and-tutorials.md) | What each paper deposited, its licence and size, and how it becomes a packaged tutorial |
| — | [References](references.md) | Every paper and dataset, with DOIs |

Operational documentation (running the pipeline, GPU hosts, the DGX Spark container) is in
[../running.md](../running.md).
