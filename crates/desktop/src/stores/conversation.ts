import { defineStore } from "pinia";
import { ref, computed } from "vue";
import { invoke } from "@tauri-apps/api/core";
import type {
  Conversation,
  ConversationSummary,
  Message,
} from "../types";

export const useConversationStore = defineStore("conversation", () => {
  const conversations = ref<ConversationSummary[]>([]);
  const activeConversation = ref<Conversation | null>(null);
  const streamingContent = ref("");

  const activeId = computed(() => activeConversation.value?.id ?? null);

  async function loadConversations() {
    conversations.value = await invoke<ConversationSummary[]>(
      "list_conversations"
    );
  }

  async function selectConversation(id: string) {
    activeConversation.value = await invoke<Conversation>(
      "get_conversation",
      { id }
    );
    streamingContent.value = "";
  }

  async function createConversation(systemPrompt?: string) {
    const conv = await invoke<Conversation>("create_conversation", {
      systemPrompt,
    });
    activeConversation.value = conv;
    streamingContent.value = "";
    await loadConversations();
  }

  async function deleteConversation(id: string) {
    await invoke("delete_conversation", { id });
    if (activeConversation.value?.id === id) {
      activeConversation.value = null;
      streamingContent.value = "";
    }
    await loadConversations();
  }

  async function updateTitle(id: string, title: string) {
    await invoke("update_conversation_title", { id, title });
    await loadConversations();
    if (activeConversation.value?.id === id) {
      activeConversation.value.title = title;
    }
  }

  async function addMessage(role: string, content: string, images?: string[]) {
    if (!activeConversation.value) return;
    const msg = await invoke<Message>("add_message", {
      conversationId: activeConversation.value.id,
      role,
      content,
      images: images && images.length > 0 ? images : null,
    });
    activeConversation.value.messages.push(msg);
    await loadConversations();
  }

  function appendStreamToken(text: string) {
    streamingContent.value += text;
  }

  function clearStreamingContent() {
    streamingContent.value = "";
  }

  function finalizeAssistantMessage(msg: Message) {
    if (!activeConversation.value) return;
    activeConversation.value.messages.push(msg);
    streamingContent.value = "";
  }

  return {
    conversations,
    activeConversation,
    activeId,
    streamingContent,
    loadConversations,
    selectConversation,
    createConversation,
    deleteConversation,
    updateTitle,
    addMessage,
    appendStreamToken,
    clearStreamingContent,
    finalizeAssistantMessage,
  };
});
