<script setup lang="ts">
import type { PendingImage } from "../../types";

const props = defineProps<{ image: PendingImage }>();
const emit = defineEmits<{
  remove: [id: string];
  click: [image: PendingImage];
}>();
</script>

<template>
  <div class="preview-item" @click="emit('click', props.image)">
    <div class="thumbnail-wrapper">
      <img :src="image.thumbnailUrl" class="thumbnail" draggable="false" />
      <button
        class="remove-btn"
        @click.stop="emit('remove', image.id)"
        title="Remove image"
      >
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
          <line x1="18" y1="6" x2="6" y2="18" />
          <line x1="6" y1="6" x2="18" y2="18" />
        </svg>
      </button>
    </div>
    <div class="status-bar">
      <div v-if="image.status === 'encoding'" class="progress-track">
        <div class="progress-fill" :style="{ width: image.progress + '%' }" />
      </div>
      <div v-else-if="image.status === 'ready'" class="status-ready">
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3">
          <polyline points="20 6 9 17 4 12" />
        </svg>
      </div>
      <div v-else class="status-error">!</div>
      <span class="status-label" :class="{ error: image.status === 'error' }">
        {{ image.label }}
      </span>
    </div>
  </div>
</template>

<style lang="scss" scoped>
.preview-item {
  display: flex;
  flex-direction: column;
  gap: 4px;
  flex-shrink: 0;
  cursor: pointer;
}

.thumbnail-wrapper {
  position: relative;
  width: 120px;
  height: 120px;
  border-radius: var(--radius-md);
  overflow: hidden;
  border: 1px solid var(--border-color);
}

.thumbnail {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
}

.remove-btn {
  position: absolute;
  top: 4px;
  right: 4px;
  width: 22px;
  height: 22px;
  border-radius: var(--radius-full);
  background: rgba(0, 0, 0, 0.6);
  color: white;
  display: flex;
  align-items: center;
  justify-content: center;
  opacity: 0;
  transition: opacity var(--transition-fast);

  &:hover {
    background: rgba(220, 40, 40, 0.8);
  }
}

.thumbnail-wrapper:hover .remove-btn {
  opacity: 1;
}

.status-bar {
  display: flex;
  align-items: center;
  gap: 4px;
  width: 120px;
}

.progress-track {
  flex: 1;
  height: 3px;
  background: var(--border-color);
  border-radius: 2px;
  overflow: hidden;
}

.progress-fill {
  height: 100%;
  background: var(--accent);
  border-radius: 2px;
  transition: width 200ms ease;
}

.status-ready {
  color: var(--success);
  display: flex;
  align-items: center;
}

.status-error {
  color: var(--error);
  font-weight: bold;
  font-size: var(--font-size-xs);
}

.status-label {
  font-size: var(--font-size-xs);
  color: var(--text-muted);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;

  &.error {
    color: var(--error);
  }
}
</style>
