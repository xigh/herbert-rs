<script setup lang="ts">
import { useConversationStore } from "../../stores/conversation";
import { useModelStore } from "../../stores/model";

const conversationStore = useConversationStore();
const modelStore = useModelStore();

async function newChat() {
  await conversationStore.createConversation(
    modelStore.settings.default_system_prompt
  );
}
</script>

<template>
  <div class="sidebar-header">
    <span class="logo">Herbert</span>
    <button class="new-chat-btn" @click="newChat" title="New Chat">
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <line x1="12" y1="5" x2="12" y2="19" />
        <line x1="5" y1="12" x2="19" y2="12" />
      </svg>
    </button>
  </div>
</template>

<style lang="scss" scoped>
.sidebar-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 16px;
  height: var(--header-height);
  border-bottom: 1px solid var(--border-color);
}

.logo {
  font-size: var(--font-size-lg);
  font-weight: 600;
  color: var(--text-primary);
}

.new-chat-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 32px;
  height: 32px;
  border-radius: var(--radius-md);
  color: var(--text-secondary);
  transition: all var(--transition-fast);

  &:hover {
    background: var(--bg-hover);
    color: var(--text-primary);
  }
}
</style>
