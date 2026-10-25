<script lang="ts">
  import { Layers } from '@lucide/svelte';
  export let selected = 0;
</script>

<details class="studio-options lora-controls">
  <summary><Layers size={16} />LoRA adapter <span>{selected > 1 ? `${selected} selected` : selected ? 'Selected' : 'None'}</span></summary>
  <div class="lora-group-body"><slot /></div>
</details>

<style>
  .lora-controls.studio-options { border-bottom: 1px solid var(--line); }
  .lora-group-body { display: grid; gap: 12px; padding-bottom: 16px; }
  .lora-group-body :global(.lora-fields) { display: grid; grid-template-columns: minmax(0, 1fr) 120px; gap: 12px; }
  .lora-group-body :global(.lora-path) { display: flex; gap: 8px; }
  .lora-group-body :global(.lora-path input) { min-width: 0; flex: 1; }
  .lora-group-body :global(.lora-path button) { flex-shrink: 0; }
  .lora-group-body :global(label) { margin-top: 0; }
  .lora-group-body :global(p) { margin: 0; color: var(--muted); font-size: 12px; line-height: 1.5; }
  .lora-group-body :global(small) { color: var(--muted); font-size: 11px; line-height: 1.5; }
  .lora-group-body :global(button) { justify-self: start; }
  @media (max-width: 600px) { .lora-group-body :global(.lora-fields) { grid-template-columns: 1fr; } }
</style>
