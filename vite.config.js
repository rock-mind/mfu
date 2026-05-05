import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// IMPORTANT: replace 'llm-training-calculator' below with your actual GitHub repo name.
// If you deploy to https://<user>.github.io/<repo>/, the base must be '/<repo>/'.
// If you deploy to a custom domain or user/organization site, set base to '/'.
export default defineConfig({
  plugins: [react()],
  base: '/mfu/',
});
