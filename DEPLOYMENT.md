# Deployment Guide — Render Web Service

This guide provides end-to-end instructions for deploying the **Hybrid CNN-LSTM-GCN Crime Prediction System** to [Render](https://render.com) using its free web tier.

---

## 1. Prerequisites
1. A [GitHub](https://github.com) account.
2. A free [Render](https://render.com) account.
3. Your local repository pushed to a GitHub repository (public or private).

---

## 2. Push Code to GitHub

Before deploying, make sure your latest code and deployment configuration files are committed and pushed:

```bash
git add .
git commit -m "Configure project for Render deployment with WhiteNoise and Gunicorn"
git push origin main
```

*(Note: `.gitignore` automatically prevents your local virtual environment, SQLite database, and `.env` files from being committed).*

---

## 3. Option A: One-Click Blueprint Deployment (Recommended)

Because this repository includes a [`render.yaml`](render.yaml) file, you can deploy the entire stack automatically:

1. Log into your **Render Dashboard**: [dashboard.render.com](https://dashboard.render.com).
2. Click **New +** in the top navigation bar and select **Blueprint**.
3. Connect your GitHub account and select your **`Hybrid_CNN_LSTM_Crime_Prediction`** repository.
4. Render will read `render.yaml` and configure the service automatically.
5. Click **Apply**. Render will start the build and deployment process.

---

## 4. Option B: Manual Web Service Setup

If you prefer to configure the Web Service manually in the Render dashboard:

1. In the Render Dashboard, click **New +** > **Web Service**.
2. Select **Build and deploy from a Git repository** > Click **Next**.
3. Choose your repository and click **Connect**.
4. Configure the service settings:
   * **Name**: `crime-prediction-system` *(or your preferred name)*
   * **Region**: `Oregon (US West)` *(or closest to you)*
   * **Branch**: `main`
   * **Root Directory**: *(Leave blank)*
   * **Runtime**: `Python 3`
   * **Build Command**: `./build.sh`
   * **Start Command**: `gunicorn crime_prediction_web.wsgi:application --bind 0.0.0.0:$PORT --workers 2 --threads 4 --timeout 120`
   * **Instance Type**: `Free`

---

## 5. Environment Variables in Render Dashboard

In your Render service page, navigate to the **Environment** tab and add the following Environment Variables:

| Key | Value | Description |
|:---|:---|:---|
| `PYTHON_VERSION` | `3.10.12` | Pins the Python runtime version. |
| `DEBUG` | `False` | Disables development debug mode in production. |
| `SECRET_KEY` | *(Click "Generate" or paste a 50+ char random string)* | Production cryptographic signing key. |
| `ALLOWED_HOSTS` | `.onrender.com,localhost,127.0.0.1` | Allows Render domain and subdomains. |
| `CSRF_TRUSTED_ORIGINS` | `https://*.onrender.com` | Prevents CSRF errors on HTTPS form submissions. |

*(Optional: If using Render PostgreSQL, Render will automatically inject `DATABASE_URL` and Django will automatically connect to it).*

---

## 6. Build & Deployment Lifecycle

When Render starts the build, `build.sh` executes the following sequence:
1. **Dependency Installation**: `pip install -r requirements.txt`
2. **Static Asset Collection**: `python manage.py collectstatic --no-input` (compresses and hashes 155+ static files into `staticfiles/` via WhiteNoise).
3. **Database Migration**: `python manage.py migrate` (applies Django authentication and content-type tables).
4. **Server Startup**: Gunicorn boots with 2 workers and 4 threads to handle concurrent web requests.

---

## 7. Post-Deployment Verification Checklist

Once Render displays **`Live`**, open your free subdomain (`https://<your-service-name>.onrender.com`) and verify the following pages:

1. **Home Page (`/`)**:
   - Verify that the hero section coordinate grid visual and dual metric chips (`96.47%` & `96.10%`) render cleanly.
   - Verify styling in both Light and Dark modes.
2. **Prediction Page (`/prediction/`)**:
   - Select a State (e.g., `MAHARASHTRA`) and District (e.g., `MUMBAI`).
   - Click **Run Multi-View Prediction** and verify that predictions, risk scores, and the GCN spatial topology graph display properly.
3. **Interactive Geo Map (`/dataset/`)**:
   - Verify that the India map loads with district boundaries and hover details.
4. **Performance Page (`/performance/`)**:
   - Verify the 5-Fold Cross-Validation charts and 300-point residual scatter plots render without errors.
5. **Static Assets**:
   - Open Browser Developer Tools (`F12` > Network) and confirm that all CSS, JS, and font files return `HTTP 200` or `HTTP 304` via WhiteNoise.

---

## 8. Free Tier Operational Notes
* **Cold Starts**: On Render's Free tier, the service spins down after 15 minutes of inactivity. When a user visits the URL after spin-down, the first request may take ~40–50 seconds to wake up the container. Subsequent requests will be fast.
* **Database**: By default, SQLite (`db.sqlite3`) is used, which is lightweight and optimal for demonstration/review without requiring an external database instance.
