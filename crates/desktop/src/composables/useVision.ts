import { ref, type Ref } from "vue";
import { invoke, Channel } from "@tauri-apps/api/core";
import type { PendingImage, VisionEvent } from "../types";

const pendingImages: Ref<PendingImage[]> = ref([]);
const lightboxImage: Ref<PendingImage | null> = ref(null);

let idCounter = 0;

export function useVision() {
  async function addImages(paths: string[]) {
    const imageExts = [".jpg", ".jpeg", ".png", ".webp"];
    const filtered = paths.filter((p) => {
      const lower = p.toLowerCase();
      return imageExts.some((ext) => lower.endsWith(ext));
    });

    for (const path of filtered) {
      const id = `img_${Date.now()}_${idCounter++}`;

      let thumbnailUrl = "";
      try {
        thumbnailUrl = await invoke<string>("read_image_thumbnail", { path });
      } catch (e) {
        console.warn("Thumbnail failed, using placeholder:", e);
      }

      const img: PendingImage = {
        id,
        thumbnailUrl,
        path,
        status: "encoding",
        progress: 0,
        label: "Queued...",
      };
      pendingImages.value.push(img);

      // Start encoding
      encodeImage(id, path);
    }
  }

  async function encodeImage(imageId: string, path: string) {
    const channel = new Channel<VisionEvent>();

    const t0 = performance.now();
    channel.onmessage = (event: VisionEvent) => {
      const elapsed = (performance.now() - t0).toFixed(0);
      const img = pendingImages.value.find((i) => i.id === imageId);
      if (!img) return;

      switch (event.type) {
        case "progress":
          console.log(`[vision-ts] +${elapsed}ms  ${imageId} progress ${event.percent}% "${event.label}"`);
          img.progress = event.percent;
          img.label = event.label;
          break;
        case "done":
          console.log(`[vision-ts] +${elapsed}ms  ${imageId} DONE ${event.num_tokens} tokens`);
          img.status = "ready";
          img.progress = 100;
          img.label = "Ready";
          img.numTokens = event.num_tokens;
          break;
        case "error":
          console.log(`[vision-ts] +${elapsed}ms  ${imageId} ERROR: ${event.message}`);
          img.status = "error";
          img.label = event.message;
          break;
      }
    };

    try {
      await invoke("encode_image", { imageId, path, channel });
    } catch (e) {
      const img = pendingImages.value.find((i) => i.id === imageId);
      if (img && img.status === "encoding") {
        img.status = "error";
        img.label = String(e);
      }
    }
  }

  async function removeImage(id: string) {
    try {
      await invoke("remove_image", { imageId: id });
    } catch {
      // ignore
    }
    pendingImages.value = pendingImages.value.filter((i) => i.id !== id);
    if (lightboxImage.value?.id === id) {
      lightboxImage.value = null;
    }
  }

  function clearImages() {
    for (const img of pendingImages.value) {
      if (img.status === "encoding") {
        invoke("remove_image", { imageId: img.id }).catch(() => {});
      }
    }
    pendingImages.value = [];
    lightboxImage.value = null;
  }

  function getReadyImageIds(): string[] {
    return pendingImages.value
      .filter((i) => i.status === "ready")
      .map((i) => i.id);
  }

  function getReadyImages(): { id: string; thumbnailUrl: string }[] {
    return pendingImages.value
      .filter((i) => i.status === "ready")
      .map((i) => ({ id: i.id, thumbnailUrl: i.thumbnailUrl }));
  }

  function openLightbox(img: PendingImage) {
    lightboxImage.value = img;
  }

  function closeLightbox() {
    lightboxImage.value = null;
  }

  return {
    pendingImages,
    lightboxImage,
    addImages,
    removeImage,
    clearImages,
    getReadyImageIds,
    getReadyImages,
    openLightbox,
    closeLightbox,
  };
}
