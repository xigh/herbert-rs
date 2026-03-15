<script setup lang="ts">
import { useModelStore } from "../stores/model";

const modelStore = useModelStore();
</script>

<template>
  <div class="loading-overlay">
    <div class="loading-content">
      <div class="spinner"></div>
      <p class="step">{{ modelStore.loadingStep || "Loading model..." }}</p>
      <div class="progress-bar-container">
        <div
          class="progress-bar-fill"
          :style="{ width: modelStore.loadingPercent + '%' }"
        ></div>
      </div>
      <p class="percent">{{ modelStore.loadingPercent }}%</p>
    </div>
  </div>
</template>

<style lang="scss" scoped>
.loading-overlay {
  position: fixed;
  inset: 0;
  background: var(--bg-overlay);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 200;
}

.loading-content {
  text-align: center;
  color: var(--text-primary);
  width: 360px;
}

.step {
  margin-top: 16px;
  font-size: var(--font-size-base);
  color: var(--text-secondary);
}

.progress-bar-container {
  margin-top: 16px;
  height: 6px;
  background: var(--bg-input);
  border-radius: var(--radius-full);
  overflow: hidden;
}

.progress-bar-fill {
  height: 100%;
  background: var(--accent);
  border-radius: var(--radius-full);
  transition: width 0.3s ease;
  min-width: 2%;
}

.percent {
  margin-top: 8px;
  font-size: var(--font-size-sm);
  color: var(--text-muted);
  font-variant-numeric: tabular-nums;
}

.spinner {
  width: 48px;
  height: 48px;
  border: 3px solid var(--border-color);
  border-top-color: var(--accent);
  border-radius: 50%;
  margin: 0 auto;
  animation: spin 0.8s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}
</style>
