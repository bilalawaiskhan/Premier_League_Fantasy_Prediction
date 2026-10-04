# Hosting this app

Recommended hobby setup: host the Next.js frontend on Vercel and the FastAPI Docker service on Render. Netlify also supports Next.js, but this repository's backend is Python. Vercel currently documents Python/FastAPI Functions as Beta; keeping the API on a regular Docker web service avoids restructuring this repository around that runtime.

## 1. Deploy the API on Render

1. Sign in to Render and choose **New → Web Service**.
2. Connect `bilalawaiskhan/Premier_League_Fantasy_Prediction`, branch `main`.
3. Select **Docker** as the runtime. Leave **Root Directory** blank so Docker can read `data/`, `src/`, and `outputs/` from the repository root.
4. Set **Dockerfile Path** to `backend/Dockerfile` and **Docker Build Context** to `.`.
5. Choose the **Free** plan for a demo, or a paid plan if you need an always-on API and a persistent disk.
6. Set health check path to `/api/health`.
7. Add environment variable `FPL_PROJECT_ROOT=/app`. `CORS_ORIGINS` can be set after the frontend has its public URL.
8. Create the service and wait for its deploy to finish. Copy its URL, such as `https://fpl-ai-api.onrender.com`.
9. Check `https://YOUR-API.onrender.com/api/health` and confirm the response status is `ok` and `live_model_available` is `true`.

The Docker image includes the model and historical CSV files. The Docker command reads Render's `PORT` environment variable.

## 2. Deploy the frontend on Vercel

1. Sign in to Vercel and choose **Add New → Project**.
2. Import `bilalawaiskhan/Premier_League_Fantasy_Prediction` from GitHub.
3. Set **Root Directory** to `frontend` and keep the detected Next.js framework/build settings.
4. Add environment variable `NEXT_PUBLIC_API_URL=https://YOUR-API.onrender.com` using the API URL from step 1, with no trailing slash.
5. Deploy and copy the Vercel public URL, such as `https://premier-league-fantasy-prediction.vercel.app`.

## 3. Allow the frontend to call the API

In Render, open the API service's **Environment** settings. Set `CORS_ORIGINS` to the exact Vercel origin, for example `https://premier-league-fantasy-prediction.vercel.app` (no path and no trailing slash), then save and redeploy. If you later add a custom domain, include its origin too, separated by a comma.

Open the Vercel site and confirm the API status changes to connected. Then try **Build my team**, wait for the first live prediction fetch to finish, and use **Auto-pick best squad**.

## Free-tier behavior and persistence

Render Free web services sleep after 15 minutes without a request and may take about a minute to wake. Their filesystem is ephemeral: the API cache database is lost on sleep/redeploy, so after waking the backend may need to fetch player histories again before forecasts are ready. A paid Render service can attach a persistent disk. To store only the FPL HTTP cache there, mount a disk at `/var/data` and set `FPL_API_CACHE_PATH=/var/data/fpl_api_cache.sqlite`. Do not mount the disk over `/app/outputs`, because that directory contains the saved models and reports. LocalStorage squads remain in each visitor's browser and are not shared between devices.

No FPL API key is needed. Review each host's current usage and pricing pages before selecting a paid plan. References: [Vercel Next.js](https://vercel.com/frameworks/nextjs), [Vercel Python runtime](https://vercel.com/docs/functions/runtimes/python), [Render Docker](https://render.com/docs/docker), [Render free services](https://render.com/docs/free), [Render persistent disks](https://render.com/docs/disks).
