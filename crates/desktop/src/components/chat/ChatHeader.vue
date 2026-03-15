<script setup lang="ts">
import { ref } from "vue";
import { invoke } from "@tauri-apps/api/core";
import { save } from "@tauri-apps/plugin-dialog";
import { useConversationStore } from "../../stores/conversation";
import { useModelStore } from "../../stores/model";

const conversationStore = useConversationStore();
const modelStore = useModelStore();

const exportStatus = ref<"" | "ok" | "error">("");

async function downloadJson() {
  const conv = conversationStore.activeConversation;
  if (!conv) return;
  try {
    const path = await save({
      defaultPath: `${conv.title || conv.id}.json`,
      filters: [{ name: "JSON", extensions: ["json"] }],
    });
    if (!path) return; // user cancelled
    await invoke("export_conversation", { id: conv.id, path });
    exportStatus.value = "ok";
    setTimeout(() => (exportStatus.value = ""), 2000);
  } catch (e) {
    console.error("Export failed:", e);
    exportStatus.value = "error";
    setTimeout(() => (exportStatus.value = ""), 2000);
  }
}
</script>

<template>
  <div class="chat-header">
    <div class="header-left">
      <h2 class="title">{{ conversationStore.activeConversation?.title }}</h2>
    </div>
    <div class="header-right">
      <button
        v-if="conversationStore.activeConversation"
        class="download-btn"
        :class="{ ok: exportStatus === 'ok', error: exportStatus === 'error' }"
        @click="downloadJson"
        title="Export to Downloads folder"
      >
        <svg v-if="exportStatus === ''" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/>
          <polyline points="7 10 12 15 17 10"/>
          <line x1="12" y1="15" x2="12" y2="3"/>
        </svg>
        <svg v-else-if="exportStatus === 'ok'" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <polyline points="20 6 9 17 4 12"/>
        </svg>
        <svg v-else width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <line x1="18" y1="6" x2="6" y2="18"/>
          <line x1="6" y1="6" x2="18" y2="18"/>
        </svg>
      </button>
      <span v-if="modelStore.modelInfo" class="model-badge">
        {{ modelStore.modelInfo.name }}
      </span>
      <span v-if="modelStore.generating" class="generating-badge">
        Generating...
      </span>
    </div>
  </div>
</template>

<style lang="scss" scoped>
.chat-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 24px;
  height: var(--header-height);
  border-bottom: 1px solid var(--border-color);
  flex-shrink: 0;
}

.title {
  font-size: var(--font-size-lg);
  font-weight: 500;
  color: var(--text-primary);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.header-right {
  display: flex;
  align-items: center;
  gap: 8px;
}

.download-btn {
  background: none;
  border: 1px solid var(--border-color);
  color: var(--text-secondary);
  cursor: pointer;
  padding: 5px 8px;
  border-radius: var(--radius-md);
  display: flex;
  align-items: center;
  transition: color 0.2s, border-color 0.2s;

  &:hover {
    color: var(--text-primary);
    border-color: var(--text-muted);
  }

  &.ok {
    color: var(--success);
    border-color: var(--success);
  }

  &.error {
    color: var(--error);
    border-color: var(--error);
  }
}

.model-badge {
  font-size: var(--font-size-xs);
  padding: 3px 8px;
  border-radius: var(--radius-full);
  background: var(--accent-light);
  color: var(--accent);
}

.generating-badge {
  font-size: var(--font-size-xs);
  padding: 3px 8px;
  border-radius: var(--radius-full);
  background: rgba(234, 179, 8, 0.15);
  color: var(--warning);
  animation: pulse 1.5s ease-in-out infinite;
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}
</style>
