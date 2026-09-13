# Interview Preparation: Face Recognition Identification System

Everything here matches the code and the **measured** results in `results/`. Numbers you can quote:

| Fact | Value | Where it comes from |
|---|---|---|
| Model | MTCNN detector + FaceNet (InceptionResnetV1, VGGFace2), 512-D | `src/detector.py`, `src/embedder.py` |
| Thresholds | accept **0.71**, uncertain **0.61**, max **3** attempts | `src/config.py`, `results/lfw_val/calibration.json` |
| Enrolled people correctly recognized (1 photo / after re-verification) | **89.6% / 97.3%** | `results/lfw_test/report.md` |
| Unknown people falsely accepted (1 photo / after re-verification) | **0.43% / 1.14%** | same |
| Old threshold 0.60 would have accepted | **7.5%** of unknown people (test) | `results/lfw_test/threshold_sweep.csv` |
| Group test: strangers falsely accepted | **0 of 300** (both scenarios) | `results/group_test/summary.json` |
| LFW standard benchmark (1:1 verification) | **99.30% ± 0.32%** (published: 99.65%) | `results/lfw_pairs/summary.json` |
| Tests | 85 passing (71 unit + 10 integration + 4 UI) | `pytest` |

---

## 1. 60-second explanation

"I built a face recognition system that can enroll people and then identify them in new photos, for free and on a CPU.
First MTCNN finds every face and its eye positions. I align and crop each face and pass it through a pretrained FaceNet
model, which turns the face into a 512-number embedding. At enrollment I store those embeddings. At identification I
compare a new face's embedding with every stored one using cosine similarity and take the best match. The key point is
that the best match is only a *candidate*: it is accepted only if the similarity is at least 0.71. Below 0.61 the answer
is UNKNOWN. In between, the result is UNCERTAIN and the system asks for another photo, at most three times, and after that
it says 'unable to verify'. I chose the thresholds on one set of LFW people and measured the results on a completely
different set: about 90% of enrolled people are recognized from one photo, 97% after re-verification, and false acceptance
is 0.43% per photo."

## 2. 2-minute technical explanation

"The project is split into small modules. `load_image` validates the file, applies the phone's EXIF rotation and converts to
RGB. `FaceDetector` runs MTCNN, a cascade of three small CNNs that outputs boxes and five landmarks. `align_and_crop` rotates
the face so the eyes are level, crops it, resizes it to 160×160 and standardises it with (x − 127.5)/128, which is what the
VGGFace2 FaceNet model was trained with. `FaceEmbedder` runs InceptionResnetV1 and L2-normalises the output, so each face is a
unit vector in 512 dimensions.

For enrollment each photo must contain exactly one good-quality face. I reject duplicates, photos that don't match the
person's other photos, and faces that already belong to another ID. Each accepted photo is stored as its own reference
embedding in a JSON file; I never store the photos.

For identification, the recognizer multiplies the matrix of all reference embeddings by the query embedding. Because
everything is unit length, that gives the cosine similarity to every reference. Each person's score is their best
reference. Then a simple policy decides: ≥ 0.71 recognized, < 0.61 unknown, in between uncertain. Uncertain starts a
`VerificationSession` that allows at most three photos, freezes the thresholds, and ends as RECOGNIZED, UNKNOWN or
UNABLE_TO_VERIFY.

I didn't pick the thresholds by hand. I built identity-disjoint validation and test splits from LFW. On validation I chose the
lowest accept threshold that keeps false acceptance at or below 1%, even after the three-attempt workflow, and the uncertain
threshold so that at most 2% of genuine users are rejected outright. Then I reported on the test split. I also ran the standard
LFW benchmark: 99.3%, close to the published 99.65%, which tells me the preprocessing is implemented correctly."

## 3. Complete data flow

```
upload -> load_image (validate, EXIF rotate, RGB, <=1600 px)
       -> MTCNN: boxes, probabilities, 5 landmarks (keep prob >= 0.90)
       -> align_and_crop: level the eyes, crop, 160x160, (x-127.5)/128      [3x160x160 tensor]
       -> InceptionResnetV1 -> 512 floats -> L2 normalise                  [unit vector]
ENROLL:   quality gates -> duplicate check -> same-person check -> other-ID check -> database.json
IDENTIFY: references (n x 512) @ query (512) -> cosine similarities
          -> best per person -> best overall = candidate
          -> decide(similarity, 0.71, 0.61) -> RECOGNIZED / UNCERTAIN / UNKNOWN
          -> UNCERTAIN: VerificationSession, up to 3 photos -> RECOGNIZED / UNKNOWN / UNABLE_TO_VERIFY
```

## 4. Every major component

| Component | What it does | One line to say |
|---|---|---|
| `utils.load_image` | validates format/size, EXIF rotation, RGB, downscale | "Bad files become friendly errors, and phone photos aren't sideways." |
| `FaceDetector` (MTCNN) | finds faces + landmarks | "Detection answers *where*, not *who*." |
| `align_and_crop` | levels the eyes, crops, standardises | "Same geometry and normalisation the model was trained with." |
| `FaceEmbedder` (FaceNet) | face → 512-D unit vector | "A learned fingerprint of the face." |
| `FaceDatabase` | JSON store, atomic writes, validation | "Stores embeddings, never photos; refuses to overwrite a corrupt file." |
| `FaceRecognizer` | cosine similarity, best match, decision | "The closest person is only a candidate." |
| `VerificationSession` | bounded re-verification | "At most 3 attempts, then stop." |
| `FaceRecognitionSystem` | enrollment + identification pipeline | "One code path, so enrollment and identification preprocess identically." |
| `evaluation.py` | metrics, calibration, session simulation | "Thresholds chosen on one set of people, reported on another." |

## 5. Why this model

- It was already in the project, and I **verified** it instead of replacing it: 99.3% on the standard LFW pairs
  (published figure 99.65%).
- Free, pip-installable, CPU-friendly, no C++ build tools on Windows.
- InsightFace/ArcFace is stronger, but its pretrained models are for non-commercial research only and it's harder to install on
  Windows. It's my first suggested improvement.

## 6. Why embeddings

Raw pixels change completely with lighting, pose and background, so comparing pixels doesn't work. The network was trained
on millions of faces to map the same person to nearby vectors and different people to distant ones. An embedding is also
small (512 floats), fast to compare, and enrolling a new person needs no retraining: we just store their vector.

## 7. Why cosine similarity

The embeddings are trained so that identity is in the **direction** of the vector. Cosine similarity measures the angle and
ignores length, so it's the natural comparison. It ranges from −1 to 1, which is easy to read. With unit-length vectors it's
just a dot product, so comparing against every reference is one matrix multiplication. For unit vectors, Euclidean distance
gives the same ranking: ‖a − b‖² = 2 − 2·cos.

## 8. Why a threshold is necessary

Without one, the system always answers with *someone*: the nearest enrolled person, even for a stranger. The threshold is
what makes it an **open-set** system that can say "I don't know this person".

## 9. UNKNOWN rejection

If the best similarity is below 0.61, the answer is UNKNOWN whoever the closest person is. Example from my tests: Rahul 0.48,
Anjali 0.41, Priya 0.35 with threshold 0.60 → UNKNOWN. On the LFW test split, 93.8% of never-enrolled people were rejected
from a single photo, and the group test had 0 of 300 strangers accepted.

## 10. UNCERTAIN state

Scores between 0.61 and 0.71 are where genuine and impostor scores overlap: 8.5% of genuine photos and 5.8% of unknown photos
land there. Accepting them would let in look-alikes, and rejecting them would turn away real users. So the system says
"UNCERTAIN, please give me another photo". UNCERTAIN is **added** to UNKNOWN; it doesn't replace it.

## 11. Re-verification

The new photo goes through the **complete** pipeline again (detection, embedding, matching). No manual override. The
thresholds are frozen for the session. A photo with no face or several faces is refused but doesn't use up an attempt.
Re-uploading the same photo is allowed, but it gives the same score because the model is deterministic, so a different photo
is more useful.

## 12. Maximum attempts

Three by default (configurable). Each completed session refuses any further attempt, so an infinite loop is impossible;
there's a unit test that tries to loop forever. Three UNCERTAIN results end as **UNABLE TO VERIFY**, which is treated as
*not recognized*. Why 3: on the test split, 97% of genuine sessions were resolved within 2 attempts, and every extra
attempt is another chance for an impostor.

## 13. False acceptance vs. false rejection

- **FAR (false acceptance rate):** a stranger is accepted as an enrolled person. It's a security failure.
- **FRR (false rejection rate):** an enrolled person is rejected. It's a usability failure.
- Raising the threshold lowers FAR and raises FRR. At 0.60 the test FAR was 7.5%; at 0.71 it's 0.43%, but single-photo FRR is ~10%.
- The UNCERTAIN band plus re-verification recovers most of that FRR (97.3% recognized after re-verification) at the cost of
  a little FAR (0.43% → 1.14%). **I measured that trade-off rather than claiming it's free.**

## 14. Multiple faces

Every face is detected, embedded and decided **independently**, with its own box, similarity and state. In the group test
(A enrolled, B/C/D not, in 100 composite images) all 400 people were detected, 0 strangers were accepted, and A was never
labelled as someone else. Enrollment photos with more than one face are rejected: in LFW, "just take the largest face"
picked the wrong person in 3.4% of photos. Limitation: re-verification handles one UNCERTAIN face at a time.

## 15. Evaluation

- LFW, split by **person** into validation and test (no one in both). 152 enrolled people per split, ~450 genuine photos,
  1,612 photos of 1,153 never-enrolled people.
- Thresholds chosen on validation, **reported on test**.
- Metrics: recognition rate, mis-identification, UNCERTAIN, FRR, unknown rejection, FAR, EER (3.2% at 0.64), threshold
  sweep, confusion matrix, score histograms, and simulated re-verification sessions.
- Honest result: session FAR on test was 1.14%, slightly above the 1% target that validation met. I reported it instead of
  re-tuning on the test set.

## 16. Failure cases I actually saw

- Look-alikes: an unenrolled man scored 0.77 against an enrolled politician → false acceptance.
- The top impostor scores were concentrated in one demographic group, consistent with known bias in face recognition. I
  didn't do a formal fairness study.
- Big appearance change (film role, different hair and make-up) → 0.29 → false rejection. Extreme expression 0.55, a hat over the
  forehead 0.58 → false rejections.
- The old 0.60 threshold accepted a bystander in a Colin Powell photo at 0.68.
- A likely LFW labelling error (a photo scores 0.87 against another person's gallery).

## 17. Limitations

Calibrated on LFW with 152 people, not on the target camera. FAR grows with gallery size. No liveness detection (a printed
photo works). Embeddings aren't encrypted. The quality checks are simple heuristics. The database is a linear scan (fine for
thousands of references).

## 18. Future improvements

ArcFace embeddings; calibration on in-domain data; liveness detection; a learned face-quality score; score normalisation for
large galleries; a fairness evaluation on a balanced dataset; encrypted storage; FAISS search; requiring the same candidate
across re-verification attempts.

## 19. Deployment

Package it in Docker with pinned requirements and the weights baked in. Put the recognizer behind a small FastAPI service with
the database in PostgreSQL + pgvector (or FAISS). Add HTTPS, authentication, audit logs and monitoring of score
distributions (drift). Keep a GPU optional: CPU takes ~0.12 s per photo here. Re-calibrate the thresholds on data from the
real cameras before go-live, and again whenever the camera or the user population changes.

## 20. Biometric privacy and security

Embeddings are biometric data: get consent (the app requires it), store only embeddings, allow deletion, encrypt at rest,
restrict access, set a retention period, and follow the law (India's DPDP Act 2023, GDPR). Never commit photos or
embeddings to GitHub (`.gitignore` excludes `data/` and `embeddings/`). Treat UNCERTAIN / UNABLE TO VERIFY as "not
recognized", and don't use the output as the only basis for important decisions.

---

## Likely interviewer questions: short answers

**Why embeddings instead of raw images?**
Pixels change with light, pose and background. Embeddings are learned so the same person stays close and different people
stay far apart. They are also tiny and let me add a person without retraining.

**Why cosine similarity?**
Identity is encoded in the direction of the vector. Cosine compares directions, ranges from −1 to 1, and for unit vectors it's
just a dot product, so it's fast.

**How did you choose the threshold?**
From data. On a validation split of LFW I picked the lowest accept threshold with false acceptance ≤ 1%, including after the
three-attempt re-verification, and the uncertain threshold so at most 2% of real users get rejected outright. That gave 0.71
and 0.61. I then measured them on different people. They're starting values, not universal: re-calibrate on your own data.

**What happens if the person is unknown?**
Their best similarity is usually low (the average is ~0.49 on LFW), below 0.61, so the answer is UNKNOWN. If they happen to
land in the band they get UNCERTAIN, and their re-verification photos usually end in UNKNOWN or UNABLE TO VERIFY.

**Why can the highest similarity still be UNKNOWN?**
There is always *some* closest person, even for a stranger. Being closest doesn't mean being the same person; the score has
to clear the threshold.

**What happens when similarity is borderline?**
It's UNCERTAIN: not accepted, not rejected. The system asks for another photo and runs the full pipeline again.

**Why re-verification?**
Borderline genuine users are often just a bad photo away from a clear score. In my test, recognition went from 89% to 97%.
The cost is more chances for an impostor, which I measured (0.43% → 1.14% FAR).

**Why a maximum of 3 attempts?**
To avoid an endless loop and to limit the impostor's chances. Most genuine cases are resolved by the second attempt.

**What if all 3 attempts are uncertain?**
The session ends as UNABLE TO VERIFY, which is treated as not recognized. The user must start a new session.

**What happens with two or more faces?**
Each face gets its own embedding, similarity and decision. In enrollment, multi-face photos are rejected because the
system shouldn't guess which face belongs to the person.

**What happens under poor lighting?**
The detector may miss the face, or the embedding shifts and the score drops, so usually a false rejection or UNCERTAIN, not a
false acceptance. The app warns about dark or blurry faces and refuses them at enrollment.

**Why use a pretrained model?**
Training a face model needs millions of labelled faces and lots of GPU time. The pretrained model already extracts good face
features; I only run inference.

**What are FAR and FRR?**
FAR: strangers accepted (a security risk). FRR: genuine users rejected (a usability problem). The threshold trades one against the other.

**How would you improve the system?**
ArcFace embeddings, calibration on real camera data, liveness detection, a learned quality score, encrypted storage,
FAISS for scale, and a fairness evaluation.

**What are the limitations?**
LFW-calibrated thresholds, FAR grows with gallery size, no anti-spoofing, plain-text embeddings, simple quality heuristics.

**How would you deploy it?**
Docker + FastAPI + a vector database, with HTTPS, authentication, logging, drift monitoring, and re-calibration on-site.

**How would you protect biometric data?**
Consent, store embeddings only, encryption at rest, access control, a retention/deletion policy, never commit biometric data,
and compliance with the DPDP Act / GDPR.
