<script setup lang="ts">
import { ref, onMounted, onUnmounted } from "vue";
import { getCurrentWindow } from "@tauri-apps/api/window";
import ChatHeader from "./ChatHeader.vue";
import MessageList from "./MessageList.vue";
import ChatInput from "./ChatInput.vue";
import ImageDropOverlay from "../vision/ImageDropOverlay.vue";
import ImagePreviewPanel from "../vision/ImagePreviewPanel.vue";
import ImageLightbox from "../vision/ImageLightbox.vue";
import { useVision } from "../../composables/useVision";

const {
  pendingImages,
  lightboxImage,
  addImages,
  removeImage,
  openLightbox,
  closeLightbox,
} = useVision();

const isDragging = ref(false);
let unlisten: (() => void) | null = null;

onMounted(async () => {
  const appWindow = getCurrentWindow();
  unlisten = await appWindow.onDragDropEvent((event) => {
    switch (event.payload.type) {
      case "enter":
        isDragging.value = true;
        break;
      case "over":
        break;
      case "drop":
        isDragging.value = false;
        addImages(event.payload.paths);
        break;
      case "leave":
        isDragging.value = false;
        break;
    }
  });
});

onUnmounted(() => {
  if (unlisten) unlisten();
});
</script>

<template>
  <div class="chat-view">
    <ChatHeader />
    <MessageList />
    <ImagePreviewPanel
      :images="pendingImages"
      @remove="removeImage"
      @preview="openLightbox"
    />
    <ChatInput />
    <ImageDropOverlay :visible="isDragging" />
    <ImageLightbox
      v-if="lightboxImage"
      :image="lightboxImage"
      @close="closeLightbox"
    />
  </div>
</template>

<style lang="scss" scoped>
.chat-view {
  display: flex;
  flex-direction: column;
  height: 100%;
  min-height: 0;
}
</style>
