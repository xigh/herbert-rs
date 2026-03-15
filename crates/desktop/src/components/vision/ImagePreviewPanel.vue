<script setup lang="ts">
import type { PendingImage } from "../../types";
import ImagePreviewItem from "./ImagePreviewItem.vue";

defineProps<{ images: PendingImage[] }>();
const emit = defineEmits<{
  remove: [id: string];
  preview: [image: PendingImage];
}>();
</script>

<template>
  <div v-if="images.length > 0" class="preview-panel">
    <div class="preview-scroll">
      <ImagePreviewItem
        v-for="img in images"
        :key="img.id"
        :image="img"
        @remove="emit('remove', $event)"
        @click="emit('preview', $event)"
      />
    </div>
  </div>
</template>

<style lang="scss" scoped>
.preview-panel {
  padding: 8px 24px;
  flex-shrink: 0;
}

.preview-scroll {
  max-width: 768px;
  margin: 0 auto;
  display: flex;
  gap: 8px;
  overflow-x: auto;
  padding-bottom: 4px;

  &::-webkit-scrollbar {
    height: 4px;
  }
  &::-webkit-scrollbar-track {
    background: transparent;
  }
  &::-webkit-scrollbar-thumb {
    background: var(--border-color);
    border-radius: 2px;
  }
}
</style>
