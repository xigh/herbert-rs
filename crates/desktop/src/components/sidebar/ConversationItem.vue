<script setup lang="ts">
import type { ConversationSummary } from "../../types";

defineProps<{
  conversation: ConversationSummary;
  active: boolean;
}>();

const emit = defineEmits<{
  select: [];
  delete: [];
}>();

function formatDate(dateStr: string): string {
  const d = new Date(dateStr);
  const now = new Date();
  const diff = now.getTime() - d.getTime();
  const days = Math.floor(diff / (1000 * 60 * 60 * 24));
  if (days === 0) return "Today";
  if (days === 1) return "Yesterday";
  if (days < 7) return `${days}d ago`;
  return d.toLocaleDateString();
}
</script>

<template>
  <div
    class="conversation-item"
    :class="{ active }"
    @click="emit('select')"
  >
    <div class="item-content">
      <div class="title">{{ conversation.title }}</div>
      <div class="meta">
        <span class="date">{{ formatDate(conversation.updated_at) }}</span>
        <span class="count">{{ conversation.message_count }} msgs</span>
      </div>
    </div>
    <button
      class="delete-btn"
      @click.stop="emit('delete')"
      title="Delete conversation"
    >
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <line x1="18" y1="6" x2="6" y2="18" />
        <line x1="6" y1="6" x2="18" y2="18" />
      </svg>
    </button>
  </div>
</template>

<style lang="scss" scoped>
.conversation-item {
  display: flex;
  align-items: center;
  padding: 10px 12px;
  border-radius: var(--radius-md);
  cursor: pointer;
  transition: background var(--transition-fast);
  gap: 8px;

  &:hover {
    background: var(--bg-hover);

    .delete-btn {
      opacity: 1;
    }
  }

  &.active {
    background: var(--bg-active);
  }
}

.item-content {
  flex: 1;
  min-width: 0;
}

.title {
  font-size: var(--font-size-sm);
  color: var(--text-primary);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.meta {
  display: flex;
  gap: 8px;
  font-size: var(--font-size-xs);
  color: var(--text-muted);
  margin-top: 2px;
}

.delete-btn {
  opacity: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  width: 24px;
  height: 24px;
  border-radius: var(--radius-sm);
  color: var(--text-muted);
  transition: all var(--transition-fast);
  flex-shrink: 0;

  &:hover {
    color: var(--error);
    background: rgba(239, 68, 68, 0.1);
  }
}
</style>
