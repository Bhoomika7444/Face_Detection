# data/

Face photos are **biometric data** and are never committed: everything in this folder except this file is git-ignored.

## Layout expected by the evaluation (`python -m src.evaluation --data data`)

```
data/
  enrolled/<person_id>/*.jpg        photos used to enroll each person (3+ recommended)
  test/known/<person_id>/*.jpg      OTHER photos of the same enrolled people  -> should be RECOGNIZED
  test/unknown/<any_id>/*.jpg       people who are NOT enrolled               -> should be UNKNOWN
```

- A `test/known` person must also exist in `enrolled/`. A `test/unknown` person must not.
- Use one person per photo (if a photo has several faces, the face nearest the centre is used).
- Only use photos of people who have agreed to it.

`python scripts/prepare_lfw.py` creates `data/lfw_eval/val` and `data/lfw_eval/test` in this layout from the public LFW dataset.
