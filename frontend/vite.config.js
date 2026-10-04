import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In dev, /api/* is proxied to the FastAPI backend so the browser never needs CORS.
// In Docker, nginx does the same proxying (see nginx.conf).
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: process.env.VITE_BACKEND_URL || "http://localhost:8000",
        changeOrigin: true,
        ws: true, // /api/ws/voice
        rewrite: (p) => p.replace(/^\/api/, ""),
      },
    },
  },
});
