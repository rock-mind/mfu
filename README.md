# LLM Training Calculator

Estimate pre-training wall-clock for any transformer architecture, or reverse-engineer the MFU you're actually achieving from a measured step time.

Supports MLA / MHA / GQA / MQA / Sliding Window attention, dense or MoE FFN, with optional Multi-Token Prediction.

## Quick start

```bash
npm install
npm run dev
```

Open http://localhost:5173 — done.

## Build for production

```bash
npm run build
npm run preview   # preview the build locally
```

The static site goes to `dist/`.

## Deploy to GitHub Pages

### One-time setup

1. **Create the repo on GitHub** (e.g. `llm-training-calculator`).
2. **Edit `vite.config.js`** — change `base` to match your repo name:
   ```js
   base: '/<your-repo-name>/',
   ```
   If you're deploying to a user/organization root site (`<user>.github.io`), set `base: '/'` instead.
3. **Enable Pages**: in the repo, go to Settings → Pages → Source → "GitHub Actions".

### Push and deploy

```bash
git init
git add .
git commit -m "init"
git branch -M main
git remote add origin git@github.com:<you>/<repo>.git
git push -u origin main
```

The workflow at `.github/workflows/deploy.yml` will build and publish automatically on every push to `main`. Site goes live at `https://<you>.github.io/<repo>/` within ~1 minute.

### Custom domain (optional)

Add a `CNAME` file in the repo root with your domain name, then set `base: '/'` in `vite.config.js`.

## Project structure

```
.
├── index.html              # Vite entry (loads fonts, mounts root)
├── src/
│   ├── main.jsx            # React mount point
│   └── Calculator.jsx      # The full component (single-file)
├── vite.config.js          # Build config; set `base` to your repo name
├── package.json
└── .github/workflows/deploy.yml
```

## License

MIT
