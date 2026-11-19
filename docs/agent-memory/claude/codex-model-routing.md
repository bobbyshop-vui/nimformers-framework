---
name: codex-model-routing
description: Gọi Codex như sub-agent qua MCP hoặc plugin codex:codex-rescue; luna mặc định
metadata:
  node_type: memory
  type: feedback
  originSessionId: dda4b68f-9037-40a7-8307-40dea514d13a
  modified: 2026-11-17T10:20:23.618Z
---

Gọi Codex như sub-agent: ưu tiên Agent với subagent_type "codex:codex-rescue" (user muốn dạng này, không phải tool MCP mcp__codex-cli__codex; MCP còn hay timeout). Cấm codex exec. Model mặc định gpt-6-luna; gpt-6.1-sol chỉ cho việc cực khó. Chia task nhỏ, mỗi lời gọi MCP một task, tái dùng sessionId.

**Why:** User muốn tiết kiệm limit Claude.

**How to apply:** Luôn truyền model và workingDirectory tường minh.
