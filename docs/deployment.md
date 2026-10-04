# Deployment

Docker Compose is intended for local evaluation and starts PostgreSQL, FastAPI, and Next.js. The backend image includes the saved artifact, source, and historical data, so image size will include those files. For hosted deployment, build the frontend with its public backend URL and set matching backend CORS origins. Provide a PostgreSQL `DATABASE_URL` and protect secrets through the host's secret manager.

No deployment has been run or verified. Squad endpoints currently have no user authentication and must not be exposed publicly as-is. Configure HTTPS, access control, request limits, backups, and database migrations before a public launch. Vercel frontend + Render/Railway backend + managed PostgreSQL are compatible targets but require separate configuration.
