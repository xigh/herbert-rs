import { defineStore } from "pinia";
import { ref } from "vue";
import { invoke, Channel } from "@tauri-apps/api/core";
import type { ModelInfo, AppSettings, LoadingEvent } from "../types";

export const useModelStore = defineStore("model", () => {
  const modelInfo = ref<ModelInfo | null>(null);
  const loading = ref(false);
  const generating = ref(false);
  const showSettings = ref(false);
  const loadingStep = ref("");
  const loadingPercent = ref(0);
  const settings = ref<AppSettings>({
    model_path: null,
    backend: "metal-q4",
    default_system_prompt: "You are a helpful assistant.",
    default_settings: {
      temperature: 0.4,
      top_k: 40,
      top_p: 0.9,
      max_tokens: 2048,
    },
    nothink: false,
  });

  async function loadSettings() {
    settings.value = await invoke<AppSettings>("get_settings");
  }

  async function saveSettings(newSettings: AppSettings) {
    await invoke("save_settings", { settings: newSettings });
    settings.value = newSettings;
  }

  async function loadModel(modelPath: string, backend: string) {
    loading.value = true;
    loadingStep.value = "Initializing...";
    loadingPercent.value = 0;

    const channel = new Channel<LoadingEvent>();
    channel.onmessage = (event: LoadingEvent) => {
      switch (event.type) {
        case "progress":
          loadingStep.value = event.step;
          loadingPercent.value = event.percent;
          break;
        case "done":
          loadingPercent.value = 100;
          break;
        case "error":
          console.error("Loading error:", event.message);
          break;
      }
    };

    try {
      modelInfo.value = await invoke<ModelInfo>("load_model", {
        modelPath,
        backendName: backend,
        channel,
      });
    } finally {
      loading.value = false;
      loadingStep.value = "";
      loadingPercent.value = 0;
    }
  }

  async function unloadModel() {
    await invoke("unload_model");
    modelInfo.value = null;
  }

  async function checkModelInfo() {
    modelInfo.value = await invoke<ModelInfo | null>("get_model_info");
  }

  return {
    modelInfo,
    loading,
    generating,
    showSettings,
    loadingStep,
    loadingPercent,
    settings,
    loadSettings,
    saveSettings,
    loadModel,
    unloadModel,
    checkModelInfo,
  };
});
