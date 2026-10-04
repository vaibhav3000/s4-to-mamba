import { defineConfig } from "@playwright/test";

// Serves the demo statically so the suite tests the repo state, not the deployment.
export default defineConfig({
  testDir: "./tests",
  timeout: 20000,
  use: {
    baseURL: "http://127.0.0.1:8123",
  },
  webServer: {
    command: "node serve.mjs 8123",
    port: 8123,
    reuseExistingServer: true,
    timeout: 10000,
  },
});
