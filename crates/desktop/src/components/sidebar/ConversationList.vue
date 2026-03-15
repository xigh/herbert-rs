<script setup lang="ts">
import ConversationItem from "./ConversationItem.vue";
import { useConversationStore } from "../../stores/conversation";

const conversationStore = useConversationStore();
</script>

<template>
  <div class="conversation-list">
    <ConversationItem
      v-for="conv in conversationStore.conversations"
      :key="conv.id"
      :conversation="conv"
      :active="conv.id === conversationStore.activeId"
      @select="conversationStore.selectConversation(conv.id)"
      @delete="conversationStore.deleteConversation(conv.id)"
    />
    <div v-if="conversationStore.conversations.length === 0" class="empty-state">
      No conversations yet
    </div>
  </div>
</template>

<style lang="scss" scoped>
.conversation-list {
  flex: 1;
  overflow-y: auto;
  padding: 8px;
}

.empty-state {
  padding: 24px 16px;
  text-align: center;
  color: var(--text-muted);
  font-size: var(--font-size-sm);
}
</style>
