<script setup lang="ts">
import SidebarHeader from "./SidebarHeader.vue";
import ConversationList from "./ConversationList.vue";
import SidebarFooter from "./SidebarFooter.vue";
</script>

<template>
  <aside class="sidebar">
    <SidebarHeader />
    <ConversationList />
    <SidebarFooter />
  </aside>
</template>

<style lang="scss" scoped>
.sidebar {
  width: var(--sidebar-width);
  min-width: var(--sidebar-width);
  height: 100%;
  background: var(--bg-sidebar);
  border-right: 1px solid var(--border-color);
  display: flex;
  flex-direction: column;
}
</style>
