<script setup lang="ts">
import { computed, ref } from "vue";
import { Marked } from "marked";
import { markedHighlight } from "marked-highlight";
import hljs from "highlight.js";
import type { Message } from "../../types";

const props = defineProps<{
  message: Message;
  streaming?: boolean;
}>();

const marked = new Marked(
  markedHighlight({
    langPrefix: "hljs language-",
    highlight(code: string, lang: string) {
      if (lang && hljs.getLanguage(lang)) {
        return hljs.highlight(code, { language: lang }).value;
      }
      return hljs.highlightAuto(code).value;
    },
  })
);

const renderedContent = computed(() => {
  if (!props.message.content) return "";
  return marked.parse(props.message.content) as string;
});

const isUser = computed(() => props.message.role === "user");

const copied = ref(false);
async function copyContent() {
  try {
    await navigator.clipboard.writeText(props.message.content);
    copied.value = true;
    setTimeout(() => (copied.value = false), 1500);
  } catch {
    // fallback
  }
}
</script>

<template>
  <div class="message-bubble" :class="{ user: isUser, assistant: !isUser }">
    <div class="role-label">{{ isUser ? "You" : "Herbert" }}</div>
    <div v-if="message.images?.length" class="message-images">
      <img
        v-for="(src, idx) in message.images"
        :key="idx"
        :src="src"
        class="message-thumb"
      />
    </div>
    <div class="content" v-html="renderedContent"></div>
    <span v-if="streaming" class="cursor"></span>
    <div class="message-footer">
      <div v-if="message.stats" class="stats">
        <span>{{ message.stats.prefill_tokens }} prefill ({{ message.stats.prefill_ms }}ms)</span>
        <span class="sep">&middot;</span>
        <span>{{ message.stats.decode_tokens }} tokens ({{ message.stats.decode_ms }}ms)</span>
        <span class="sep">&middot;</span>
        <span>{{ message.stats.tokens_per_sec.toFixed(1) }} tok/s</span>
      </div>
      <div v-else class="stats-spacer"></div>
      <button class="copy-btn" @click="copyContent">
        <svg v-if="!copied" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <rect x="9" y="9" width="13" height="13" rx="2" ry="2"/>
          <path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"/>
        </svg>
        <svg v-else width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <polyline points="20 6 9 17 4 12"/>
        </svg>
      </button>
    </div>
  </div>
</template>

<style lang="scss" scoped>
.message-bubble {
  position: relative;
  padding: 12px 16px;
  border-radius: var(--radius-lg);
  line-height: 1.7;

  &.user {
    background: var(--bg-message-user);
    color: var(--text-on-primary);
    align-self: flex-end;
    max-width: 85%;

    .copy-btn {
      color: rgba(255, 255, 255, 0.5);
      &:hover {
        color: #ffffff;
      }
    }
  }

  &.assistant {
    background: var(--bg-message-assistant);
    color: var(--text-primary);
    max-width: 100%;
  }
}

.message-images {
  display: flex;
  gap: 6px;
  flex-wrap: wrap;
  margin-bottom: 8px;
}

.message-thumb {
  height: 80px;
  border-radius: var(--radius-md);
  object-fit: cover;
  cursor: pointer;
}

.role-label {
  font-size: var(--font-size-xs);
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  margin-bottom: 4px;
  opacity: 0.7;
}

.content {
  :deep(p) {
    margin-bottom: 8px;
    &:last-child {
      margin-bottom: 0;
    }
  }

  :deep(pre) {
    margin: 8px 0;
    background: rgba(0, 0, 0, 0.3);
    border-radius: var(--radius-md);
    padding: 12px 16px;
    overflow-x: auto;
  }

  :deep(code) {
    font-family: var(--font-mono);
    font-size: var(--font-size-sm);
  }

  :deep(ul), :deep(ol) {
    padding-left: 20px;
    margin: 8px 0;
  }

  :deep(li) {
    list-style: disc;
    margin-bottom: 4px;
  }

  :deep(ol li) {
    list-style: decimal;
  }

  :deep(blockquote) {
    border-left: 3px solid var(--border-light);
    padding-left: 12px;
    margin: 8px 0;
    color: var(--text-secondary);
  }

  :deep(a) {
    color: var(--accent);
    text-decoration: underline;
  }

  :deep(table) {
    border-collapse: collapse;
    margin: 8px 0;
    width: 100%;

    th, td {
      border: 1px solid var(--border-color);
      padding: 6px 12px;
      text-align: left;
    }

    th {
      background: rgba(0, 0, 0, 0.2);
    }
  }
}

.cursor {
  display: inline-block;
  width: 2px;
  height: 1.2em;
  background: var(--accent);
  margin-left: 2px;
  vertical-align: text-bottom;
  animation: blink 0.8s step-end infinite;
}

@keyframes blink {
  0%, 100% { opacity: 1; }
  50% { opacity: 0; }
}

.message-footer {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-top: 8px;
  gap: 8px;
}

.stats {
  font-size: var(--font-size-xs);
  color: var(--text-muted);
  cursor: default;
  font-variant-numeric: tabular-nums;
  display: flex;
  gap: 4px;
  flex-wrap: wrap;
}

.sep {
  opacity: 0.5;
}

.stats-spacer {
  flex: 1;
}

.copy-btn {
  background: none;
  border: none;
  color: var(--text-muted);
  cursor: pointer;
  padding: 4px;
  border-radius: var(--radius-sm);
  display: flex;
  align-items: center;
  opacity: 0.6;
  transition: opacity 0.2s, color 0.2s;
  flex-shrink: 0;

  &:hover {
    opacity: 1;
    color: var(--text-primary);
  }
}
</style>
