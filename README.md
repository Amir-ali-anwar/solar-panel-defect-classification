# Solar Panel Condition Classifier

Classifies a photo of a solar panel into one of six conditions: **Clean**,
**Dusty**, **Bird-drop**, **Electrical-damage**, **Physical-Damage**, or
**Snow-Covered**.

The project has three parts:

| Path        | Purpose                                                              |
|-------------|-----------------------------------------------------------------------|
| `ml/`       | Training pipeline. Produces the model artifact in `models/`.         |
| `backend/`  | FastAPI service that loads the trained model and serves predictions. |
| `frontend/` | React + Vite web app for uploading a photo and viewing the result.   |

## 1. Train the model

```bash
python -m venv .venv-ml
.venv-ml\Scripts\activate          # Windows
pip install -r ml/requirements.txt
python -m ml.train
```

This reads images from `nootebooks/Data/<class_name>/*`, does a stratified
70/15/15 train/val/test split, trains a MobileNetV2 transfer-learning model
(frozen backbone, then a fine-tuning phase), and writes to `models/`:

- `solar_panel_classifier.keras` — the trained model
- `class_names.json` — ordered list of class labels
- `metrics.json` — test-set accuracy and a full classification report
- `training_history.png`, `confusion_matrix.png` — diagnostic plots

Why transfer learning instead of the from-scratch CNN in the original
notebook: the dataset is small (~870 images across 6 classes), which is not
enough to train a convolutional network from scratch without heavy
overfitting. A frozen ImageNet backbone with a light classification head
generalizes much better on this dataset size.

Useful flags: `--epochs`, `--fine-tune-epochs`, `--batch-size`, `--skip-fine-tune`.

### Results (last run)

Trained on this machine at 224x224 with `--batch-size 4`, the full
MobileNetV2 backbone unfrozen for fine-tuning (`FINE_TUNE_AT_LAYER = 0` in
`ml/config.py`, `FINE_TUNE_LEARNING_RATE = 5e-6`), stronger augmentation
(`RandomTranslation`/`RandomBrightness` added on top of flip/rotation/
zoom/contrast), a cosine-decay LR schedule for both phases, and
oversampling of minority classes in the training split (`ml/data.py`'s
`oversample_to_balance` duplicates Physical-Damage/Electrical-damage/etc.
up to the size of the largest class, val/test are left untouched) on the
original ~885-image dataset: **85.7% test accuracy** across all 6 classes,
early-stopped at epoch 20/20 (frozen phase) and epoch 3/25 (fine-tuning).
Per-class F1 ranged from 0.71 (Physical-Damage, the smallest class at 11
test images) to 0.93 (Electrical-damage). Full breakdown in
`models/metrics.json` and `models/confusion_matrix.png`.

Run-to-run variance on this dataset size is real (dropout, augmentation,
and CPU op non-determinism aren't seeded) — we've seen anywhere from 79.7%
to 85.7% across otherwise-identical configs, so treat any single run's
number as a point estimate, not an exact benchmark.

**Aiming for 90%+:** the backend also applies test-time augmentation (TTA)
at inference — see below — on top of whatever the raw model scores.
Oversampling closed some of the gap (83.5% → 85.7%, Physical-Damage F1
0.67 → 0.71), but revealed the real shape of the remaining problem:
Physical-Damage precision is now a perfect 1.0 (zero false alarms) while
recall is still only 0.55 (misses half the real damage photos, 6/11) — the
model got more conservative about that class rather than better at
recognizing it. With only 11 test images, that's the point where
algorithmic tricks stop helping much; more *curated* real Physical-Damage
photos (not a bulk unfiltered dataset merge — see the lesson below) is the
remaining lever most likely to move this further.

**Adding more training data:** we tried merging in a large Roboflow export
(~22,000 additional images, capped/balanced down to ~4,190) and it made
every class worse (73.3% accuracy, down from 85.7% at the time) — most
likely because the new images had a different visual distribution (uniform
600x600 Roboflow preprocessing vs. the original varied raw photos) and one
class (Electrical-damage) barely grew while others grew 5-9x, worsening its
already-thin relative representation. Lesson: **always back up
`models/*.keras`/`*.json` before retraining on a materially different
dataset** — we hadn't, and lost the 85.7% weights when the worse run
overwrote them in place. `models/backups/` now holds a copy of that failed
run's artifacts for reference. If you want to try augmenting the dataset
again, a smaller, curated addition (verified images per class, not a bulk
unfiltered dump) is likely to work better than what we tried.

An earlier run at 160x160 with only the top ~50 backbone layers unfrozen
got 79.7% accuracy and noticeably lower confidence (rarely above 70%) —
the smaller input resolution and shallower fine-tuning were both throwing
away signal the model needed. If you're on a very memory-constrained
machine and have to drop back to 160x160/partial unfreeze, expect that
gap.

**Windows/CPU/low-RAM troubleshooting:** on a memory-constrained machine you
may hit two distinct failures:

- `Insufficient memory (case 4)` or an `OOM when allocating tensor` error
  even though RAM looks free: some source photos here go up to 6240x4160
  (~78MB decoded as uint8), and `ml/data.py` already downsamples during
  JPEG decode (via PIL's `draft()` mode) to keep peak memory bounded — if
  it still OOMs, your system just has little free RAM at that moment; free
  some up or lower `--batch-size` (2–4 works on ~3GB free).
- Training hangs with 0% CPU usage: don't set `TF_NUM_INTEROP_THREADS=1` —
  the image pipeline uses `tf.py_function`, which needs at least 2 inter-op
  threads to hand control back to Python, or it deadlocks.

What actually worked here:

```bash
set TF_ENABLE_ONEDNN_OPTS=0
set TF_NUM_INTRAOP_THREADS=1
set OMP_NUM_THREADS=1
python -m ml.train --batch-size 4
```

## 2. Run the backend API

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-dev.txt
cp .env.example .env   # adjust paths if needed
uvicorn app.main:app --reload
```

Endpoints (all under `/api`):

- `GET /api/health` — `{ "status": "ok", "model_loaded": true }`
- `GET /api/classes` — list of class names
- `POST /api/predict` — multipart `file` upload → predicted class, confidence, per-class probabilities

`/api/predict` uses test-time augmentation (TTA): each upload is scored as
3 views (original, horizontal flip, a 90%-center-crop zoom) and the
softmax probabilities are averaged, instead of a single forward pass. This
costs ~3x the inference time (still under 2s on CPU) in exchange for more
stable, less noisy predictions — see `ModelService._build_tta_batch` in
`backend/app/inference.py`.

Run tests: `pytest` (from `backend/`). Tests use a stub model service, so
they don't require a trained artifact.

## 3. Run the frontend

```bash
cd frontend
npm install
cp .env.example .env   # optional: set VITE_API_BASE_URL for a remote backend URL
npm run dev
```

Open http://localhost:5173, drop in a panel photo, and click **Analyze photo**.

By default, the frontend calls `/api/*` on the same origin. In local dev,
Vite proxies `/api` to `http://localhost:8000`.

## Run everything with Docker Compose

Train the model first (step 1) so `models/` contains the artifact, then:

```bash
docker compose up --build
```

- Frontend: http://localhost:8080
- Backend: http://localhost:8000

In Docker, Nginx proxies frontend `/api/*` calls to the `backend` service,
so the browser uses one origin (`:8080`) and avoids CORS complexity.

## 4. Production deployment prep

Before deploying:

- Set backend env vars (`MODEL_PATH`, `CLASS_NAMES_PATH`, `CORS_ORIGINS`, `MAX_UPLOAD_MB`).
- Ensure your host mounts `models/` read-only and contains:
  - `solar_panel_classifier.keras`
  - `class_names.json`
- For split hosting (frontend on Vercel/Netlify, backend elsewhere), set:
  - `frontend/.env`: `VITE_API_BASE_URL=https://your-backend-domain`
  - backend `CORS_ORIGINS` to include your frontend domain.

Quick smoke checks after deploy:

- `GET /api/health` returns status ok and model_loaded true.
- Upload one valid JPEG/PNG image and confirm prediction response.
- Upload an invalid file type and confirm `415 Unsupported Media Type`.

## 5. Deploy backend on Render and frontend on Vercel

### A) Deploy backend on Render

This repo now includes a Render Blueprint config at `render.yaml`.

1. Push this project to GitHub.
2. In Render, choose **New +** -> **Blueprint**.
3. Select your repository and deploy.
4. After first deploy, open the backend service settings and set:
  - `CORS_ORIGINS=["https://<your-vercel-domain>"]`
  - Optionally include preview domains too, for example:
    `CORS_ORIGINS=["https://<your-vercel-domain>","https://<your-preview-domain>"]`
5. Confirm health endpoint:
  - `https://<your-render-domain>/api/health`

### B) Deploy frontend on Vercel

This repo now includes a Vercel config at `frontend/vercel.json`.

1. In Vercel, create a new project from the same GitHub repo.
2. Set **Root Directory** to `frontend`.
3. Add environment variable:
  - `VITE_API_BASE_URL=https://<your-render-domain>`
4. Deploy.

### C) Final connection check

After both deployments:

- Open your Vercel app.
- Upload a valid image and verify prediction works.
- Verify invalid file type returns the expected API error.
- If browser requests fail with CORS, re-check `CORS_ORIGINS` on Render.

## Known limitations

- Training data (~870 images) is small and imbalanced (69–194 images per
  class); class weighting is applied during training but more data would
  improve real-world accuracy.
- The dataset (`nootebooks/Data/`, ~310MB, plus a duplicate `Data.zip`) lives
  inside the repo. For a real production setup this should move to object
  storage (S3/GCS) with only a download script committed — it was left as-is
  since that's a data-management decision rather than a code change.
- No authentication/rate-limiting on the API — add these before exposing it
  publicly.
