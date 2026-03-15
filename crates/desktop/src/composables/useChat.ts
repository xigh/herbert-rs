import { invoke, Channel } from "@tauri-apps/api/core";
import { useConversationStore } from "../stores/conversation";
import { useModelStore } from "../stores/model";
import { useVision } from "./useVision";
import type { TokenEvent } from "../types";

export function useChat() {
  const conversationStore = useConversationStore();
  const modelStore = useModelStore();
  const { getReadyImages, clearImages } = useVision();

  async function sendMessage(content: string) {
    if (!conversationStore.activeConversation) return;
    if (modelStore.generating) return;
    if (!modelStore.modelInfo) {
      throw new Error("Model not loaded");
    }

    // Collect ready image IDs and thumbnails before sending
    const readyImages = getReadyImages();
    const imageIds = readyImages.map((i) => i.id);
    const thumbnailUrls = readyImages.map((i) => i.thumbnailUrl);

    // Add user message with image thumbnails
    await conversationStore.addMessage(
      "user",
      content,
      thumbnailUrls.length > 0 ? thumbnailUrls : undefined
    );

    // Start generation
    modelStore.generating = true;
    conversationStore.clearStreamingContent();

    const channel = new Channel<TokenEvent>();

    channel.onmessage = (event: TokenEvent) => {
      switch (event.type) {
        case "delta":
          conversationStore.appendStreamToken(event.text);
          break;
        case "done":
          break;
        case "cancelled":
          break;
        case "error":
          console.error("Generation error:", event.message);
          break;
      }
    };

    try {
      await invoke("generate", {
        conversationId: conversationStore.activeConversation.id,
        imageIds: imageIds.length > 0 ? imageIds : null,
        channel,
      });

      // Reload conversation to get the saved assistant message
      if (conversationStore.activeConversation) {
        await conversationStore.selectConversation(
          conversationStore.activeConversation.id
        );
      }
      await conversationStore.loadConversations();
    } catch (e) {
      console.error("Generate error:", e);
    } finally {
      modelStore.generating = false;
      conversationStore.clearStreamingContent();
      if (imageIds.length > 0) {
        clearImages();
      }
    }
  }

  async function cancelGeneration() {
    await invoke("cancel_generation");
  }

  return {
    sendMessage,
    cancelGeneration,
  };
}
