<script setup lang="ts">
import { ref, watch, nextTick, computed } from "vue";
import MessageBubble from "./MessageBubble.vue";
import { useConversationStore } from "../../stores/conversation";
import { useModelStore } from "../../stores/model";

const conversationStore = useConversationStore();
const modelStore = useModelStore();
const scrollContainer = ref<HTMLElement | null>(null);

const messages = computed(() => conversationStore.activeConversation?.messages ?? []);
const hasStreaming = computed(
  () => modelStore.generating && conversationStore.streamingContent.length > 0
);

function scrollToBottom() {
  nextTick(() => {
    if (scrollContainer.value) {
      scrollContainer.value.scrollTop = scrollContainer.value.scrollHeight;
    }
  });
}

// Auto-scroll when new messages arrive or streaming content changes
watch(
  () => [messages.value.length, conversationStore.streamingContent],
  scrollToBottom,
  { deep: true }
);
</script>

<template>
  <div class="message-list" ref="scrollContainer">
    <div class="messages-container">
      <MessageBubble
        v-for="(msg, idx) in messages"
        :key="idx"
        :message="msg"
      />
      <!-- Streaming message -->
      <MessageBubble
        v-if="hasStreaming"
        :message="{
          role: 'assistant',
          content: conversationStore.streamingContent,
          timestamp: new Date().toISOString(),
        }"
        :streaming="true"
      />
    </div>
  </div>
</template>

<style lang="scss" scoped>
.message-list {
  flex: 1;
  overflow-y: auto;
  padding: 24px 0;
}

.messages-container {
  max-width: 768px;
  margin: 0 auto;
  padding: 0 24px;
  display: flex;
  flex-direction: column;
  gap: 16px;
}
</style>
