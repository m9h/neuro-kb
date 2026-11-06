# Intellectual property landscape (checked 2026-10-25)

Not legal advice. A claims file for the advisor to review before any protocol leaves the phantom
bench. Sources: Google Patents, Justia, company disclosures, FUS Foundation IP page.

## Filings that bear on this bench

| Holder | Filing | What it claims | Status | Bears on |
|---|---|---|---|---|
| Forest Neurotech LLC (→ Arbor / Merge Labs) | US 2025/0001217 A1 (priority 2023-06-30, pub. 2025-01-02); inventors Norman, Aflalo, Biederman, Pinton, Guan, Frick, Grisillo, Nastaskin | Ultrasound BCI that images and modulates; implantable transducer limiting local brain temp < 39 °C | Final rejection Jan 2025, response June 2025, pending | Imaging-plus-stimulation through one aperture; implants (not our scope) |
| Stanford (Nolan R. Williams, John Coetzee) | WO 2024/030928 A2 (2024-02-08); US 2026/0041941 A1 (2026-02-12), appl. 19/100,561, priority 2022-08-01 | LIFU in iTBS/cTBS-patterned trains with specified frequency, intensity, PRF, duty cycle (~1.25–3.75 % cTBS), session length; dependent claims: MDD, left DLPFC, sgACC, striatum, amygdala | Published application, pending | Any TBS-patterned LIFU protocol for depression in humans |
| Attune Neurosciences | US 11,730,960 (2023); US 12,485,297 (2025-12-02, = US 2025/0108235 A1) | Brain modulation; cranial self-registration for ultrasound neuromodulation wearables | Granted | Wearable self-registration / headset fitting |
| Openwater | 68 filings; therapeutic family: WO 2023/09196 wearable transcranial therapeutic US; prov. 63/404407 inhibition & stimulation; harmonic-excitation (oncolysis, clot lysis) | — | Mixed | **Covered by the Patent Pledge** for open-source devices on CC BY-SA designs / AGPL-type code; defensive termination |
| Butterfly Network | > 600 issued + pending (own count; 800 cited in 2023) | CMUT-on-CMOS, probes, software | Granted/pending; Fujifilm Sonosite suit (2023, 7 patents) | Not usable: Embedded licence only; Merge holds BCI R&D exclusivity |
| Insightec / Lotus Neuro | Exablate MRgFUS portfolio; BBB opening | Ablation; microbubble BBB | Granted | Adjacent (HIFU/BBB), not LIFU neuromodulation |
| Fondazione IRCCS Besta / INRiM | US 12,599,785 B2 (2026-04-14) | Patient-worn stereotactic LIFU frame | Granted | Wearable frames |
| Nudge | none found under that name | — | possibly unpublished / other entity | — |
| Merge Labs | none found under that name | — | Forest family + Butterfly licence + Shapiro/Caltech molecular ultrasound | Expect continuations and new filings |
| Others | BrainSonix, NaviFUS, NeuroFUS/Sonic Concepts, Soterix/ANT, Cordance, Carthera, Neurosona, Sonomind, NeuroHarmonics, Huakang Chaonao, Gestala, Columbia (Konofagou) | device/method filings | — | check per protocol |

## Rules we follow

1. Phantom and bench only; publication of designs and protocols is not infringement.
2. Before any human protocol is adopted, compare its pulse pattern and indication against the Stanford
   claims, and any imaging-plus-modulation architecture against the Forest family.
3. Stay on Openwater designs where possible and keep derivatives open (CERN-OHL-S hardware, open
   software) so the Openwater Patent Pledge applies.
4. Date every release in this public repository: it is prior art from that date.
5. Do not plan around Butterfly Ultrasound-on-Chip hardware.

## Open items

- Confirm current prosecution status of US 2025/0001217 and US 2026/0041941 (Patent Center).
- Search assignees "Nudge" variants, "Merge Labs", "Arbor Neuroscience", "NVIDIA" (ultrasound) for
  filings published after 2026-10-25.
- Read the full Openwater Patent Pledge PDF (README gives a summary; it lists 68 filings but the
  table has 69 rows).
