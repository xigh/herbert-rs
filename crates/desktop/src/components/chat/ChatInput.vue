<script setup lang="ts">
import { ref } from "vue";
import { open } from "@tauri-apps/plugin-dialog";
import { useChat } from "../../composables/useChat";
import { useVision } from "../../composables/useVision";
import { useModelStore } from "../../stores/model";

const modelStore = useModelStore();
const { sendMessage, cancelGeneration } = useChat();
const { addImages } = useVision();

const input = ref("");
const textarea = ref<HTMLTextAreaElement | null>(null);

async function handleSend() {
  const text = input.value.trim();
  if (!text) return;
  if (modelStore.generating) return;
  if (!modelStore.modelInfo) {
    modelStore.showSettings = true;
    return;
  }

  input.value = "";
  resizeTextarea();
  await sendMessage(text);
}

function handleKeydown(e: KeyboardEvent) {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    handleSend();
  }
  if (e.key === "Escape" && modelStore.generating) {
    cancelGeneration();
  }
}

function resizeTextarea() {
  const el = textarea.value;
  if (!el) return;
  el.style.height = "auto";
  el.style.height = Math.min(el.scrollHeight, 200) + "px";
}

async function handleClip() {
  const result = await open({
    multiple: true,
    filters: [
      {
        name: "Images",
        extensions: ["jpg", "jpeg", "png", "webp"],
      },
    ],
  });
  if (result) {
    const paths = Array.isArray(result) ? result : [result];
    addImages(paths);
  }
}
</script>

<template>
  <div class="chat-input-container">
    <div class="input-wrapper">
      <button
        class="clip-btn"
        @click="handleClip"
        :disabled="modelStore.generating"
        title="Attach images"
      >
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M21.44 11.05l-9.19 9.19a6 6 0 01-8.49-8.49l9.19-9.19a4 4 0 015.66 5.66l-9.2 9.19a2 2 0 01-2.83-2.83l8.49-8.48" />
        </svg>
      </button>
      <textarea
        ref="textarea"
        v-model="input"
        class="chat-textarea"
        placeholder="Type a message... (Enter to send, Shift+Enter for newline)"
        rows="1"
        @keydown="handleKeydown"
        @input="resizeTextarea"
        :disabled="modelStore.generating"
      ></textarea>
      <div class="input-actions">
        <button
          v-if="modelStore.generating"
          class="stop-btn"
          @click="cancelGeneration"
          title="Stop generating (Esc)"
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
            <rect x="6" y="6" width="12" height="12" rx="2" />
          </svg>
          Stop
        </button>
        <button
          v-else
          class="send-btn"
          @click="handleSend"
          :disabled="!input.trim() || !modelStore.modelInfo"
          title="Send message (Enter)"
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <line x1="22" y1="2" x2="11" y2="13" />
            <polygon points="22 2 15 22 11 13 2 9 22 2" />
          </svg>
        </button>
      </div>
    </div>
  </div>
</template>

<style lang="scss" scoped>
.chat-input-container {
  padding: 16px 24px 24px;
  flex-shrink: 0;
}

.input-wrapper {
  max-width: 768px;
  margin: 0 auto;
  display: flex;
  gap: 8px;
  align-items: flex-end;
  background: var(--bg-input);
  border: 1px solid var(--border-color);
  border-radius: var(--radius-lg);
  padding: 8px 12px;
  transition: border-color var(--transition-fast);

  &:focus-within {
    border-color: var(--accent);
  }
}

.chat-textarea {
  flex: 1;
  resize: none;
  min-height: 24px;
  max-height: 200px;
  line-height: 1.5;
  padding: 4px 0;
  color: var(--text-primary);

  &::placeholder {
    color: var(--text-muted);
  }

  &:disabled {
    opacity: 0.5;
  }
}

.clip-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 32px;
  height: 32px;
  flex-shrink: 0;
  border-radius: var(--radius-md);
  color: var(--text-muted);
  transition: color var(--transition-fast);

  &:hover:not(:disabled) {
    color: var(--text-primary);
  }

  &:disabled {
    opacity: 0.3;
    cursor: not-allowed;
  }
}

.input-actions {
  display: flex;
  gap: 4px;
  flex-shrink: 0;
}

.send-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 32px;
  height: 32px;
  border-radius: var(--radius-md);
  background: var(--accent);
  color: var(--text-on-primary);
  transition: all var(--transition-fast);

  &:hover:not(:disabled) {
    background: var(--accent-hover);
  }

  &:disabled {
    opacity: 0.3;
    cursor: not-allowed;
  }
}

.stop-btn {
  display: flex;
  align-items: center;
  gap: 4px;
  padding: 4px 12px;
  border-radius: var(--radius-md);
  background: var(--error);
  color: var(--text-on-primary);
  font-size: var(--font-size-sm);
  transition: all var(--transition-fast);

  &:hover {
    opacity: 0.9;
  }
}
</style>
