import { createApp } from "vue";
import { createPinia } from "pinia";
import { listen } from "@tauri-apps/api/event";
import { invoke } from "@tauri-apps/api/core";
import html2canvas from "html2canvas";
import App from "./App.vue";
import "./styles/main.scss";

const app = createApp(App);
app.use(createPinia());
app.mount("#app");

// Screenshot tool: triggered by file /tmp/herbert-take-screenshot
listen("take-screenshot", async () => {
  try {
    const canvas = await html2canvas(document.documentElement, {
      backgroundColor: null,
      scale: 1,
    });
    const dataUrl = canvas.toDataURL("image/png");
    const base64 = dataUrl.replace(/^data:image\/png;base64,/, "");
    await invoke("save_screenshot_data", {
      data: base64,
      path: "/tmp/herbert-screenshot.png",
    });
    console.log("[screenshot] saved to /tmp/herbert-screenshot.png");
  } catch (e) {
    console.error("[screenshot] failed:", e);
  }
});
