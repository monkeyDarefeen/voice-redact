# voice-redact Examples

This folder contains example outputs demonstrating the pipeline in both voice modes.

## Input Audio

Each input file contains 2-3 sentences with 1-2 pieces of personally identifiable information:

| File | PII Contained |
|---|---|
| **test_pii_1.wav** | Name (John Mitchell) |
| **test_pii_2.wav** | Name (Sarah Johnson) + Phone (555-0147) |
| **test_pii_3.wav** | Name (David Chen) + Email |
| **test_pii_4.wav** | Phone (555-8239) |
| **test_pii_5.wav** | Name (Emily Rodriguez) + Email |

## Output Audio

Each input was processed twice — once in **original** voice mode and once in **random** voice mode. In all outputs, the PII has been replaced with synthetic data:

| # | Original Voice | Random Voice | What Changed |
|---|---|---|---|
| 1 | `test_pii_1_original.wav` | `test_pii_1_random.wav` | "John Mitchell" → synthetic name |
| 2 | `test_pii_2_original.wav` | `test_pii_2_random.wav` | Name + phone → synthetic replacements |
| 3 | `test_pii_3_original.wav` | `test_pii_3_random.wav` | Name + email → synthetic replacements |
| 4 | `test_pii_4_original.wav` | `test_pii_4_random.wav` | Phone number → synthetic number |
| 5 | `test_pii_5_original.wav` | `test_pii_5_random.wav` | Name + email → synthetic replacements |

**Key difference:**
- **Original voice** — the output sounds like the same speaker from the input audio
- **Random voice** — the output uses a randomly designed voice (different gender/age/pitch)

## Transcripts

Each `_transcript.json` file shows the three text variants:

```
original_text  → the raw transcription (contains PII)
redacted_text  → PII replaced with [NAME], [PHONE], [EMAIL] tags
synthetic_text → PII replaced with fake but realistic data
```
