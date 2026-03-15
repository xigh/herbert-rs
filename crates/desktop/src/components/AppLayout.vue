<script setup lang="ts">
import Sidebar from "./sidebar/Sidebar.vue";
import ChatView from "./chat/ChatView.vue";
import WelcomeScreen from "./WelcomeScreen.vue";
import { useConversationStore } from "../stores/conversation";
import { useModelStore } from "../stores/model";
import { onMounted } from "vue";

const conversationStore = useConversationStore();
const modelStore = useModelStore();

onMounted(async () => {
  await conversationStore.loadConversations();
  await modelStore.loadSettings();
  await modelStore.checkModelInfo();
});
</script>

<template>
  <div class="app-layout">
    <Sidebar />
    <main class="main-content">
      <ChatView v-if="conversationStore.activeConversation" />
      <WelcomeScreen v-else />
    </main>
  </div>
</template>

<style lang="scss" scoped>
.app-layout {
  display: flex;
  height: 100%;
  width: 100%;
}

.main-content {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-width: 0;
}
</style>
