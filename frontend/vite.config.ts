import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The API server sets no CORS headers, so the browser must believe the API is
// same-origin. Everything under /api is proxied to uvicorn with the prefix
// stripped, which keeps the server free of frontend concerns.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
        // A sweep run can hold the socket for minutes.
        timeout: 30 * 60 * 1000,
        proxyTimeout: 30 * 60 * 1000,
      },
    },
  },
});
