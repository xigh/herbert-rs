<script setup lang="ts">
defineProps<{ visible: boolean }>();
</script>

<template>
  <Transition name="fade">
    <div v-if="visible" class="drop-overlay">
      <div class="drop-zone">
        <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
          <polyline points="17 8 12 3 7 8" />
          <line x1="12" y1="3" x2="12" y2="15" />
        </svg>
        <span class="drop-text">Drop images here</span>
        <span class="drop-hint">JPG, PNG, WebP</span>
      </div>
    </div>
  </Transition>
</template>

<style lang="scss" scoped>
.drop-overlay {
  position: fixed;
  inset: 0;
  z-index: 100;
  display: flex;
  align-items: center;
  justify-content: center;
  background: rgba(0, 0, 0, 0.7);
  backdrop-filter: blur(4px);
}

.drop-zone {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 12px;
  padding: 48px 64px;
  border: 2px dashed var(--accent);
  border-radius: var(--radius-lg);
  color: var(--text-primary);
}

.drop-text {
  font-size: var(--font-size-xl);
  font-weight: 500;
}

.drop-hint {
  font-size: var(--font-size-sm);
  color: var(--text-muted);
}

.fade-enter-active,
.fade-leave-active {
  transition: opacity 150ms ease;
}
.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}
</style>
