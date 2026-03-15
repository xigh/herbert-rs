<script setup lang="ts">
import { useConversationStore } from "../stores/conversation";
import { useModelStore } from "../stores/model";

const conversationStore = useConversationStore();
const modelStore = useModelStore();

async function startChat() {
  if (!modelStore.modelInfo) {
    modelStore.showSettings = true;
    return;
  }
  await conversationStore.createConversation(
    modelStore.settings.default_system_prompt
  );
}
</script>

<template>
  <div class="welcome-screen">
    <div class="welcome-content">
      <h1 class="welcome-title">Herbert</h1>
      <p class="welcome-subtitle">Local LLM inference on Apple Silicon</p>

      <div class="status-card" v-if="!modelStore.modelInfo">
        <p>No model loaded. Open settings to load a model.</p>
        <button class="action-btn" @click="modelStore.showSettings = true">
          Open Settings
        </button>
      </div>

      <div class="status-card" v-else>
        <p>
          <strong>{{ modelStore.modelInfo.name }}</strong> loaded
          ({{ modelStore.modelInfo.num_layers }} layers,
          {{ modelStore.modelInfo.hidden_size }}d)
        </p>
        <button class="action-btn" @click="startChat">
          New Chat
        </button>
      </div>
    </div>
  </div>
</template>

<style lang="scss" scoped>
.welcome-screen {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
}

.welcome-content {
  text-align: center;
  max-width: 400px;
}

.welcome-title {
  font-size: 2.5rem;
  font-weight: 700;
  color: var(--text-primary);
  margin-bottom: 8px;
}

.welcome-subtitle {
  font-size: var(--font-size-lg);
  color: var(--text-secondary);
  margin-bottom: 32px;
}

.status-card {
  background: var(--bg-secondary);
  border: 1px solid var(--border-color);
  border-radius: var(--radius-lg);
  padding: 24px;

  p {
    color: var(--text-secondary);
    margin-bottom: 16px;
  }
}

.action-btn {
  padding: 10px 24px;
  background: var(--accent);
  color: var(--text-on-primary);
  border-radius: var(--radius-md);
  font-weight: 500;
  transition: background var(--transition-fast);

  &:hover {
    background: var(--accent-hover);
  }
}
</style>
