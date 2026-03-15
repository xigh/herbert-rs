<script setup lang="ts">
import { ref, onMounted } from "vue";
import { useModelStore } from "../stores/model";
import type { AppSettings } from "../types";

const emit = defineEmits<{
  close: [];
}>();

const modelStore = useModelStore();

const form = ref<AppSettings>({
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

const loadError = ref("");

onMounted(() => {
  form.value = JSON.parse(JSON.stringify(modelStore.settings));
});

async function saveAndClose() {
  await modelStore.saveSettings(form.value);
  emit("close");
}

async function loadModel() {
  if (!form.value.model_path) {
    loadError.value = "Please enter a model path";
    return;
  }
  loadError.value = "";
  try {
    await modelStore.saveSettings(form.value);
    await modelStore.loadModel(form.value.model_path, form.value.backend);
    emit("close");
  } catch (e) {
    loadError.value = String(e);
  }
}

async function unloadModel() {
  await modelStore.unloadModel();
}

function handleOverlayClick(e: MouseEvent) {
  if ((e.target as HTMLElement).classList.contains("modal-overlay")) {
    emit("close");
  }
}
</script>

<template>
  <div class="modal-overlay" @click="handleOverlayClick">
    <div class="modal">
      <div class="modal-header">
        <h2>Settings</h2>
        <button class="close-btn" @click="emit('close')">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <line x1="18" y1="6" x2="6" y2="18" />
            <line x1="6" y1="6" x2="18" y2="18" />
          </svg>
        </button>
      </div>

      <div class="modal-body">
        <!-- Model Section -->
        <section class="section">
          <h3>Model</h3>
          <div class="field">
            <label>Model Path</label>
            <input
              v-model="form.model_path"
              type="text"
              placeholder="/path/to/model/directory"
            />
          </div>
          <div class="field">
            <label>Backend</label>
            <select v-model="form.backend">
              <option value="metal-q4">Metal Q4</option>
              <option value="metal-int8">Metal Int8</option>
              <option value="metal-bf16">Metal BF16</option>
            </select>
          </div>
          <div class="field checkbox-field">
            <label>
              <input type="checkbox" v-model="form.nothink" />
              No Think (suppress thinking blocks)
            </label>
          </div>
          <div class="field-row">
            <button class="btn-primary" @click="loadModel">
              {{ modelStore.modelInfo ? "Reload Model" : "Load Model" }}
            </button>
            <button
              v-if="modelStore.modelInfo"
              class="btn-secondary"
              @click="unloadModel"
            >
              Unload
            </button>
          </div>
          <div v-if="loadError" class="error-msg">{{ loadError }}</div>
          <div v-if="modelStore.modelInfo" class="info-msg">
            Loaded: {{ modelStore.modelInfo.name }}
            ({{ modelStore.modelInfo.backend }},
            {{ modelStore.modelInfo.num_layers }} layers)
          </div>
        </section>

        <!-- Sampling Section -->
        <section class="section">
          <h3>Sampling</h3>
          <div class="field">
            <label>Temperature ({{ form.default_settings.temperature }})</label>
            <input
              v-model.number="form.default_settings.temperature"
              type="range"
              min="0"
              max="2"
              step="0.05"
            />
          </div>
          <div class="field">
            <label>Top-K ({{ form.default_settings.top_k }})</label>
            <input
              v-model.number="form.default_settings.top_k"
              type="range"
              min="0"
              max="200"
              step="1"
            />
          </div>
          <div class="field">
            <label>Top-P ({{ form.default_settings.top_p }})</label>
            <input
              v-model.number="form.default_settings.top_p"
              type="range"
              min="0"
              max="1"
              step="0.01"
            />
          </div>
          <div class="field">
            <label>Max Tokens</label>
            <input
              v-model.number="form.default_settings.max_tokens"
              type="number"
              min="1"
              max="16384"
            />
          </div>
        </section>

        <!-- System Prompt -->
        <section class="section">
          <h3>System Prompt</h3>
          <div class="field">
            <textarea
              v-model="form.default_system_prompt"
              rows="3"
              placeholder="System prompt for new conversations"
            ></textarea>
          </div>
        </section>
      </div>

      <div class="modal-footer">
        <button class="btn-secondary" @click="emit('close')">Cancel</button>
        <button class="btn-primary" @click="saveAndClose">Save</button>
      </div>
    </div>
  </div>
</template>

<style lang="scss" scoped>
.modal-overlay {
  position: fixed;
  inset: 0;
  background: var(--bg-overlay);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 100;
}

.modal {
  background: var(--bg-modal);
  border: 1px solid var(--border-color);
  border-radius: var(--radius-lg);
  width: 520px;
  max-height: 80vh;
  display: flex;
  flex-direction: column;
  box-shadow: var(--shadow-lg);
}

.modal-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 16px 20px;
  border-bottom: 1px solid var(--border-color);

  h2 {
    font-size: var(--font-size-xl);
    font-weight: 600;
  }
}

.close-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 28px;
  height: 28px;
  border-radius: var(--radius-sm);
  color: var(--text-muted);

  &:hover {
    background: var(--bg-hover);
    color: var(--text-primary);
  }
}

.modal-body {
  flex: 1;
  overflow-y: auto;
  padding: 20px;
}

.section {
  margin-bottom: 24px;

  &:last-child {
    margin-bottom: 0;
  }

  h3 {
    font-size: var(--font-size-base);
    font-weight: 600;
    margin-bottom: 12px;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-size: var(--font-size-xs);
  }
}

.field {
  margin-bottom: 12px;

  label {
    display: block;
    font-size: var(--font-size-sm);
    color: var(--text-secondary);
    margin-bottom: 4px;
  }

  input[type="text"],
  input[type="number"],
  select,
  textarea {
    width: 100%;
    padding: 8px 12px;
    background: var(--bg-input);
    border: 1px solid var(--border-color);
    border-radius: var(--radius-md);
    color: var(--text-primary);
    font-size: var(--font-size-sm);

    &:focus {
      border-color: var(--accent);
    }
  }

  input[type="range"] {
    width: 100%;
    accent-color: var(--accent);
  }

  textarea {
    resize: vertical;
    min-height: 60px;
  }

  select {
    appearance: auto;
  }
}

.checkbox-field label {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: var(--font-size-sm);
  color: var(--text-secondary);
  cursor: pointer;

  input[type="checkbox"] {
    accent-color: var(--accent);
  }
}

.field-row {
  display: flex;
  gap: 8px;
  margin-bottom: 8px;
}

.btn-primary {
  padding: 8px 16px;
  background: var(--accent);
  color: var(--text-on-primary);
  border-radius: var(--radius-md);
  font-size: var(--font-size-sm);
  font-weight: 500;

  &:hover {
    background: var(--accent-hover);
  }
}

.btn-secondary {
  padding: 8px 16px;
  background: var(--bg-input);
  border: 1px solid var(--border-color);
  color: var(--text-primary);
  border-radius: var(--radius-md);
  font-size: var(--font-size-sm);

  &:hover {
    background: var(--bg-hover);
  }
}

.error-msg {
  font-size: var(--font-size-sm);
  color: var(--error);
  margin-top: 4px;
}

.info-msg {
  font-size: var(--font-size-sm);
  color: var(--success);
  margin-top: 4px;
}

.modal-footer {
  display: flex;
  justify-content: flex-end;
  gap: 8px;
  padding: 16px 20px;
  border-top: 1px solid var(--border-color);
}
</style>
