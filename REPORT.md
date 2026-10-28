# Assignment 2 Report

- **Name**: Javier Arce Lopez
- **Student ID**: 20158
- **Email**: jarcel.ieu2022@student.ie.edu
- **Group**: BBADBA 5B

## Business problem, and the cost of being wrong

The system supports a CCTV triage desk for a US city. It samples still frames from cameras in conflictive public areas, scores each one for physical or weaponized violence, and either ignores it, sends it to an operator, or raises an alert so police can be dispatched quickly. It complements normal patrols as a first filter over camera footage; it does not replace policing or run dispatch entirely on its own.

**Who acts.** Operators review the uncertain band. Police respond when an alert is raised, either automatically at very high confidence or after an operator confirms.

The costs below are severity weights in USD, set so the desk stays conservative about missed violence. They can be read as rough stand-ins for the harm these incidents cause, but they are not actual realistic estimates of costs.

| Outcome | Cost (USD) | Justification |
|---|---:|---|
| False positive (auto-alert on a calm frame) | 5 | Partial weight for an unnecessary police dispatch, although low because police are already on patrol |
| Missed violence (auto-ignore) | 1,000 | Physical violence continues unattended; weighted heavily so the system prefers not to miss |
| Missed weaponized violence (auto-ignore) | 100,000 | Most severe omission; catastrophic relative to a false alarm |
| True positive / true negative | 0 | No error cost once the decision is correct |
| Human review | 0.50 | Operators at $50/hour reviewing 100 frames/hour |

**Asymmetry.** Missing a real incident, especially a weaponized one, is far worse than a false alarm, so the desk should not auto-ignore unless the model is very sure the frame is calm. False positives still count: the $5 represents wasted dispatch attention, and at a 0.01% base rate even a modest auto-alert rate is expensive, which is why auto-alerts stay rare. Human review is the third outcome for the uncertain middle (see the cost section).

**Production rarity.** We assume positive frames make up 0.01% of live samples and that 20% of positives are weaponized. The estimate is deliberately inflated for conflictive camera sites. It started from rough high-crime frequency references, registered-crime rates in memphis, Tenessee, with aprox. 1 violent crime every 30 minutes, and was padded to stay conservative and to cover crime the city does not currently detect on camera.

**Pilot scope.** Five cameras, one still frame every 30 seconds, day shift only (8h). At the shipped two-threshold point the middle band needs about two operators on that day shift. Night, weekend, and city-wide 24/7 coverage are future steps and are not part of what we claim can ship today.

## Dataset

**Source.** Roboflow Universe workspace `pdl-qtfdi`, project [`crime-detection-jekv6`](https://universe.roboflow.com/pdl-qtfdi/crime-detection-jekv6), version 1. We download that public version as labelled and use its labels and class names as published.

**What one image is.** A single still frame from a source video clip of a public scene. The task is image classification: each whole frame carries one of three mutually exclusive labels, `normal`, `violence`, or `weaponized`. 

**Why this dataset serves the business problem.** The desk needs to separate calm public footage from physical violence and from weaponized violence so police are sent to the right places, and those are the project's three labels. 

## What the dataset's composition implies

The Roboflow export ships with its own train/valid/test folders, but the filenames show frames from the same source video spread across those folders. Training on that export would put near-duplicate frames of one clip in both train and test and inflate accuracy. We therefore re-split locally, still using the version 1 images and labels. We parse a video id from each filename (`n…` / `v…` / `w…`), keep every frame of a video in one split, and assign videos 70% / 15% / 15% within each class (`seed=0`). Image counts follow from the videos each split receives, so they are not forced to 70/15/15.

| Split | normal | violence | weaponized | Total images | Videos (n/v/w) |
|---|---:|---:|---:|---:|---|
| train | 2,865 | 3,386 | 2,568 | 8,819 | 140 / 71 / 70 |
| valid | 736 | 691 | 539 | 1,966 | 30 / 14 / 15 |
| test | 601 | 708 | 522 | 1,831 | 30 / 14 / 15 |

- Class balance is usable, since no class is down to a handful of images. Video counts for violence and weaponized are thin on valid and test (14 to 15 clips each), so per-class test metrics are noisy at the clip level even when image counts look large.
- The test set is about 67% positive (violence plus weaponized), while live CCTV under our production assumption is about 0.01% positive. Accuracy on this test set does not transfer to production, so cost and threshold choices must be reweighted with the production base rate.

## Finetuning and baselines

**Model and task.** `yolo11n-cls` (YOLO11 nano classification), fine-tuned on the video-level split. Classification matches the dataset's three whole-image labels, and the nano size keeps the pilot light enough for a desk dashboard. Training ran on a Google Colab GPU (it has too heavy to run it locally): 15 epochs, `imgsz=224`, `batch=16`, `seed=0`. Validation top-1 peaked around epoch 4 (~0.75) and drifted down to ~0.69 by epoch 15. We ship the best validation checkpoint instead of the last epoch, and discuss the overfitting under Limitations.

The desk's positive score is `p_violence + p_weaponized`. Both incident classes count as positive; their false-negative costs differ later.

Results on the same test split (1,831 frames):

| System | 3-class accuracy | Binary precision | Binary recall | Binary accuracy |
|---|---:|---:|---:|---:|
| Fine-tuned `yolo11n-cls` | **0.652** | **0.900** | **0.764** | **0.784** |
| Trivial baseline (always `violence`, train majority) | 0.387 | 0.672 | 1.000 | 0.672 |
| Zero-shot stand-in / predict-nothing (always `normal`) | 0.328 | 0.000 | 0.000 | 0.328 |

**Baselines.** The majority-class baseline always predicts `violence`. No zero-shot model fits this taxonomy: a pretrained `yolo11n-cls` ImageNet head and a COCO detector have no `normal` / `violence` / `weaponized` outputs, so they cannot raise our incident alerts. The closest zero-shot stand-in is therefore to predict nothing (always `normal`). The fine-tuned model beats both baselines on 3-class accuracy and on binary accuracy. Its binary lead over the majority baseline is modest (0.78 vs 0.67) because the test set is incident-heavy. The useful gain is that the model says `normal` with a specificity of 0.83, where the majority baseline alerts on everything.

## Per-class evaluation and failure modes

Confusion matrix on the test set (rows = true, columns = predicted; argmax label):

| true \ pred | normal | violence | weaponized |
|---|---:|---:|---:|
| normal | 496 | 68 | 37 |
| violence | 180 | 433 | 95 |
| weaponized | 110 | 148 | 264 |

| Class | Precision | Recall | Support |
|---|---:|---:|---:|
| normal | 0.63 | 0.83 | 601 |
| violence | 0.67 | 0.61 | 708 |
| weaponized | 0.67 | **0.51** | 522 |

Binary precision (~0.90) looks strong because, on this test mix, the model is usually right when it picks an incident class. Weaponized recall (~0.51) is the weak point: at argmax the model misses almost half of weaponized frames, sending them to `violence` (148) or `normal` (110). Some confusion between violence and weaponized is expected, since both are incident scenes, and it costs less under the desk score because both count as positive. Weaponized frames predicted as `normal` are the costly error, because they risk auto-ignore. Normal is the best-recalled class; calm frames are relatively easy here. 

## Cost matrix and operating point

Costs and production mix come from the business section: FP $5, FN_violence $1,000, FN_weaponized $100,000, review $0.50, P(positive) 0.01%, with 20% of positives weaponized. Expected cost is production-reweighted: class-conditional error rates come from the test set, and the prior comes from the assumed live mix. The single-threshold sweep is saved in `artifacts/cost_vs_threshold.png` and `artifacts/cost_sweep.json`.

**Single-threshold reference.** If the desk only auto-alerts or auto-clears (no review), cost bottoms out at t\* ≈ 0.975, about $1,016 per 1,000 frames. Always-alert costs about $5,000 per 1k and always-clear about $2,080 per 1k. At a 0.01% base rate even a modest false-positive rate is expensive next to the rare true incidents, which is why that machine-only minimum sits so high.

**Shipped two-threshold policy.** We do not ship the empty-band argmin. Defaults are `t_clear = 0.01` and `t_alert = t\* ≈ 0.975`:

- score &lt; 0.01 → auto-clear (only very confident calm frames)
- score ≥ 0.975 → auto-alert (same high bar as the machine-only minimum)
- otherwise → human review

That keeps auto-alerts rare, sharply cuts auto-ignores of real incidents relative to model-alone, and sends the uncertain middle to operators. Under production reweighting the band costs about **$666 per 1,000 frames** (~$514 error + ~$151 review).

| Quantity | Value |
|---|---|
| Expected cost | **~$666 / 1,000 frames** |
| of which errors | ~$514 / 1k |
| of which review | ~$151 / 1k |
| Review share (production-reweighted) | **~30.3%** |
| Auto-FPR (calm → alert) | ~7.8% |
| Auto-miss violence | ~5.8% |
| Auto-miss weaponized | ~5.9% |

## What a person still has to check

Frames scoring below 0.01 are ignored, frames at or above 0.975 trigger an auto-alert, and scores in between go to an operator. Under the shipped defaults that middle band is about 30% of production frames.

**Pilot load (day shift).** 5 cameras × 1 frame / 30 s × 8 h = 4,800 frames per day. At ~30.3% review that is about 1,450 looks / day, or about 1.8 people on 8h shifts (~2 operators), assuming 100 reviews/person-hour. Police still act on auto-alerts and on incidents operators confirm.

Expected USD per 1,000 frames for the shipped policy and the alternatives, under the same assumptions:

| Policy | USD / 1k frames | Staffing note |
|---|---:|---|
| Do nothing / always clear | ~2,080 | No desk; pays severity-weighted miss cost on rare positives |
| Person checks every frame | 500 | ~6 people on this pilot's day shift (4,800 looks) |
| Model alone at cost argmin (t\*≈0.975) | ~1,016 | 0 review staff; beats do-nothing; misses more incidents automatically |
| **Shipped band (t_clear=0.01, t_alert≈0.975)** | **~$666** | **~2 operators**; cheaper than model-alone and do-nothing; still above full human on pure cost-efficiency |

**Ship call for this pilot.** We ship the wide review band as complementary day-shift automation on five cameras. It is cheaper than doing nothing or letting the model decide alone, and it needs a small operator desk. 

## Limitations & next steps

1. Day shift only, so nights and weekends are uncovered. Next step: staff a second shift, or accept higher miss risk off-hours with a narrower camera set.
2. Domain shift. Test footage is curated violence clips rather than raw municipal CCTV, and its prevalence (~67% positive) is far from production (0.01%). Next step: collect a small labelled sample from the real cameras and recalibrate thresholds before widening the pilot.
3. Weaponized recall is weak (~0.51 at argmax). Next step: more weaponized clips from new videos (more frames from the same videos add little), and/or a weapon detector head if the city needs that distinction to choose the dispatch type.
4. Video-level test sets are thin (14 to 15 incident clips). Next step: more held-out videos before treating per-class gaps as stable.
5. Sampling one frame every 30 s can miss short events between samples. Next step: raise sampling on high-risk cameras only, or move to a lightweight video model once the desk process is proven.
6. Overfitting. Validation accuracy peaked early, and we ship `best.pt`. Next step: stronger augmentation and an early-stopping policy tied to the cost metric instead of top-1 alone.


## Generative AI use disclosure

Ideation, topic research and dataset research is my own production. AI has been used to implement the technical solution, analyse the tradeoffs of different implementations, and build the gradio dashboard. Conclusions, decisions and the final report are all my own production. AI was also used to organise the repository and the initial structure of the report in order to ensure no details were missed.
