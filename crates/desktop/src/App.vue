<script setup lang="ts">
import AppLayout from "./components/AppLayout.vue";
import ModelLoadingOverlay from "./components/ModelLoadingOverlay.vue";
import SettingsModal from "./components/SettingsModal.vue";
import { useModelStore } from "./stores/model";

const modelStore = useModelStore();
</script>

<template>
  <AppLayout />
  <ModelLoadingOverlay v-if="modelStore.loading" />
  <SettingsModal
    v-if="modelStore.showSettings"
    @close="modelStore.showSettings = false"
  />
</template>
