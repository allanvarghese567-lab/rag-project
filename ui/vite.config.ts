import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// base must match GitHub Pages repo name when deploying to project pages
export default defineConfig({
  plugins: [react()],
  base: "/rag-project/",
});
