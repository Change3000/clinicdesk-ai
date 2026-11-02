# ClinicDesk AI

AI triage for a GP clinic's patient inbox. Every incoming message (email, SMS, web form) is:

1. **Checked for emergencies** – red-flag rules plus a learned urgency model. A hit stops all automation: no drafted reply, staff call the patient now, and the patient is told to call 000 (or Lifeline 13 11 14).
2. **Sorted into one of 9 request types** – booking, change/cancel, repeat prescription, test results, referral, medical certificate, billing & Medicare, general enquiry, clinical question.
3. **Mined for details** – who it's for, doctor, medicines, tests, specialist, dates/times, phone number; Medicare numbers are masked.
4. **Routed and answered** – sent to the right team with a suggested response time and a drafted reply that staff approve before sending. Low-confidence messages go to a person.

Everything runs in the browser: no server, no API key, no message leaves the device.

**Live demo:** https://change3000.github.io/clinicdesk-ai/

Every push runs a GitHub Action (`.github/workflows/pages.yml`) that retrains both models from the datasets in Python, scores the emergency rules, builds the single-file app and deploys it to GitHub Pages. The training log for each run is on the **Actions** tab.

## Results (synthetic data)

| Measure | Result |
|---|---|
| Request-type accuracy, test set (90 messages) | 95.6% |
| Request-type accuracy, 27 messages written after the model was built | 88.9% |
| Auto-routed messages that were correct (test + holdout) | 97 of 99 (98%); 17 unsure messages sent to a person |
| Emergencies caught in never-seen wording | 8 of 10 |
| Routine messages wrongly flagged as emergencies | 1 of 117 |

Iteration history and what each change did is in `docs/` and on the app's **How it works** tab.

## Project layout

```
data/       training, test, holdout and safety datasets (synthetic, written for this project)
            normalise.json      SMS shorthand dictionary ("appt" -> appointment, "script" -> prescription)
            safety_rules.json   emergency red-flag rules (v3)
model/      train.py            intent classifier: TF-IDF + logistic regression, versions v1-v4, exports app/model.json
            train_urgency.py    learned urgency model (word + character n-grams), exports app/urgency.json
            test_safety.py      scores the red-flag rules on every safety set
app/        engine.js           triage engine (runs in browser and Node)
            template.html       user interface
build.py    inlines models + engine into dist/index.html (a single file that works offline)
docs/       ClinicDesk_AI_Prompt_Documentation.pdf   prompts, revisions and results
.github/    workflows/pages.yml  train -> test -> build -> deploy
```

## Reproduce

```bash
pip install scikit-learn numpy scipy
python model/train.py --version v1      # baseline
python model/train.py --version v2 --export
python model/train_urgency.py 0.5 --export
python model/test_safety.py
python build.py                         # -> dist/index.html
```

## Limits

Trained on synthetic messages, so accuracy on a real clinic's inbox will be lower until it is retrained on de-identified real messages. English only. It is an admin tool, not a medical device: it never gives clinical advice and every reply needs staff approval. The emergency check still misses some wording (2 of 10 in the last unseen test), so it supports reception staff rather than replacing them; a production version should add an LLM safety check.

Built by Vishwa Singh with Claude (Anthropic) as the AI pair-programmer. Fictional clinic ("Harbourside Family Practice"); phone numbers are from Australia's range reserved for fiction.
