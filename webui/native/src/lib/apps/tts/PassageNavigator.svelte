<script lang="ts">
  import { Plus, ChevronLeft, ChevronRight, ListFilter } from '@lucide/svelte';
  import { chunkReady, assignedSpeaker, selectedTake, type TtsProject } from './project';
  export let project: TtsProject;
  export let selectedId: string;
  export let disabled = false;
  export let preview: string[] | null = null;
  export let previewIndex = 0;
  export let checked = new Set<string>();
  export let onselect: (id: string) => void;
  export let onpreview: (index: number) => void;
  export let onadd: () => void;
  export let ongenerate: (ids: string[]) => void;
  let search = '';
  let filter = 'all';
  let page = 0;
  let bulk = false;
  const pageSize = 40;
  $: rows = project.chunks.filter(
    (chunk) =>
      chunk.text.toLowerCase().includes(search.toLowerCase()) &&
      (filter === 'all' ||
        (filter === 'missing' && !selectedTake(chunk)) ||
        (filter === 'changed' && !chunkReady(project, chunk)))
  );
  $: pages = Math.max(1, Math.ceil(rows.length / pageSize));
  $: if (page >= pages) page = pages - 1;
  $: selectedPosition = rows.findIndex((chunk) => chunk.id === selectedId);
  $: if (selectedPosition >= 0) page = Math.floor(selectedPosition / pageSize);
  function toggle(id: string, value: boolean) {
    const next = new Set(checked);
    if (value) next.add(id);
    else next.delete(id);
    checked = next;
  }
</script>

<section class="passage-navigator" aria-label="Passages">
  <header>
    <h3>Passages <span>{preview?.length ?? project.chunks.length}</span></h3>
    {#if project.chunks.length && !preview}<button
        class="icon"
        aria-label="Select multiple passages"
        aria-pressed={bulk}
        on:click={() => (bulk = !bulk)}><ListFilter size={15} /></button
      >{/if}
  </header>
  {#if preview}<p class="rail-help">Review each passage before continuing.</p>
  {:else if project.chunks.length}<div class="search-controls">
      <input
        aria-label="Search passages"
        placeholder="Find a passage…"
        bind:value={search}
        on:input={() => (page = 0)}
      /><select aria-label="Filter passages" bind:value={filter} on:change={() => (page = 0)}
        ><option value="all">All passages</option><option value="missing">Without audio</option><option value="changed"
          >Needs attention</option
        ></select
      >
    </div>{/if}
  <div class="passage-list" aria-label="Passage list">
    {#if preview}{#each preview as text, index}<button
          class="passage-row"
          class:current={previewIndex === index}
          aria-label={`Review passage ${index + 1}`}
          aria-current={previewIndex === index ? 'true' : undefined}
          on:click={() => onpreview(index)}
          ><span class="number">{String(index + 1).padStart(2, '0')}</span><span class="passage-copy"
            ><span class="excerpt">{text}</span><small>Preview</small></span
          ></button
        >{/each}
    {:else}{#each rows.slice(page * pageSize, (page + 1) * pageSize) as chunk (chunk.id)}{@const index =
          project.chunks.indexOf(chunk)}{@const ready = chunkReady(project, chunk)}
        <div class="passage-item" class:current={selectedId === chunk.id}>
          {#if bulk}<input
              class="bulk-checkbox"
              aria-label={`Select passage ${index + 1}`}
              type="checkbox"
              {disabled}
              checked={checked.has(chunk.id)}
              on:change={(e) => toggle(chunk.id, e.currentTarget.checked)}
            />{/if}<button
            class="passage-row"
            aria-label={`Edit passage ${index + 1}`}
            aria-current={selectedId === chunk.id ? 'true' : undefined}
            {disabled}
            on:click={() => onselect(chunk.id)}
            ><span class="number">{String(index + 1).padStart(2, '0')}</span><span class="passage-copy"
              ><span class="excerpt">{chunk.text || 'Empty passage'}</span><small
                ><span class="readiness" class:ready></span>{assignedSpeaker(project, chunk)?.name || 'No speaker'} · {ready
                  ? 'Ready'
                  : selectedTake(chunk)
                    ? 'Needs update'
                    : 'No audio'}</small
              ></span
            ></button
          >
        </div>{/each}
      {#if !project.chunks.length}<div class="empty-outline">
          <span class="empty-number">01</span>
          <h4>Your script, in passages</h4>
          <p>Write or import your text, then review how it splits here.</p>
        </div>{:else if !rows.length}<p class="rail-help">No matching passages. Try another search or filter.</p>{/if}
    {/if}
  </div>
  {#if !preview && project.chunks.length}<footer>
      {#if pages > 1}<div class="pagination">
          <button aria-label="Previous passage page" disabled={page === 0} on:click={() => page--}
            ><ChevronLeft size={14} /></button
          ><span>{page + 1} / {pages}</span><button
            aria-label="Next passage page"
            disabled={page === pages - 1}
            on:click={() => page++}><ChevronRight size={14} /></button
          >
        </div>{/if}{#if bulk}<label class="select-all"
          ><input
            type="checkbox"
            aria-label="Select all matching passages"
            checked={rows.length > 0 && rows.every((c) => checked.has(c.id))}
            {disabled}
            on:change={(e) => {
              const next = new Set(checked);
              for (const c of rows) e.currentTarget.checked ? next.add(c.id) : next.delete(c.id);
              checked = next;
            }}
          />Select all matches</label
        ><button disabled={disabled || !checked.size} on:click={() => ongenerate([...checked])}
          >Generate selected ({checked.size})</button
        >{/if}<button class="add-passage" {disabled} on:click={onadd}><Plus size={14} />Add passage</button>
    </footer>{/if}
</section>

<style>
  .passage-navigator {
    display: flex;
    flex-direction: column;
    height: 100%;
    min-height: 0;
  }
  header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 10px;
    padding-bottom: 18px;
    flex-shrink: 0;
  }
  h3 {
    font-size: 15px;
    margin: 0;
  }
  h3 > span {
    color: var(--muted);
    font-size: 11px;
    font-weight: 400;
    margin-left: 6px;
  }
  button {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    min-height: 32px;
    font-size: 11px;
  }
  .icon {
    padding: 5px;
    background: transparent;
    border: 0;
  }
  .icon[aria-pressed='true'] {
    color: var(--cyan);
  }
  .search-controls {
    display: grid;
    gap: 8px;
    margin-bottom: 16px;
    flex-shrink: 0;
  }
  .search-controls input,
  .search-controls select {
    width: 100%;
    min-width: 0;
    font-size: 11px;
  }
  .passage-list {
    overflow: auto;
    min-height: 0;
    flex: 1;
    scrollbar-width: thin;
  }
  .passage-row {
    width: 100%;
    display: flex;
    justify-content: flex-start;
    align-items: start;
    gap: 10px;
    padding: 14px 8px;
    border: 0;
    border-radius: 0;
    background: transparent;
    text-align: left;
    font-weight: 400;
  }
  .passage-item {
    display: flex;
    align-items: center;
    border-left: 2px solid transparent;
    border-bottom: 1px solid var(--line);
  }
  .passage-item.current,
  .passage-row.current {
    background: var(--active-bg);
    border-left-color: var(--cyan);
  }
  .number {
    color: var(--muted);
    font-size: 10px;
    flex-shrink: 0;
    padding-top: 2px;
  }
  .passage-copy {
    min-width: 0;
    line-height: 1.65;
    font-size: 12px;
  }
  .excerpt {
    display: -webkit-box;
    -webkit-line-clamp: 3;
    line-clamp: 3;
    -webkit-box-orient: vertical;
    overflow: hidden;
  }
  .passage-copy small {
    display: flex;
    align-items: center;
    font-size: 9px;
    color: var(--muted);
    margin-top: 8px;
    gap: 4px;
    line-height: 1.5;
  }
  .passage-row:hover .number,
  .passage-row:hover .passage-copy small {
    color: var(--text);
  }
  .readiness {
    display: inline-block;
    width: 5px;
    height: 5px;
    border-radius: 50%;
    background: var(--muted);
    flex-shrink: 0;
  }
  .readiness.ready {
    background: var(--cyan);
  }
  .bulk-checkbox {
    width: 14px;
    height: 14px;
    margin-left: 8px;
    flex-shrink: 0;
  }
  .rail-help {
    margin: 0 0 16px;
    font-size: 11px;
    line-height: 1.6;
    color: var(--muted);
  }
  .empty-outline {
    display: grid;
    gap: 14px;
    padding: 28px 4px;
  }
  .empty-number {
    color: var(--muted);
    font-size: 24px;
    font-weight: 300;
    opacity: 0.5;
  }
  h4 {
    margin: 0;
    font-size: 13px;
    font-weight: 500;
  }
  .empty-outline p {
    font-size: 12px;
    color: var(--muted);
    line-height: 1.7;
    margin: 0;
  }
  footer {
    display: grid;
    gap: 10px;
    padding-top: 14px;
    border-top: 1px solid var(--line);
    flex-shrink: 0;
  }
  .pagination {
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-size: 11px;
    color: var(--muted);
  }
  .pagination button {
    padding: 4px;
    border: 0;
    background: transparent;
  }
  .add-passage {
    width: 100%;
    background: transparent;
  }
  .select-all {
    display: flex;
    align-items: center;
    gap: 6px;
    color: var(--muted);
    font-size: 11px;
  }
  .select-all input {
    width: 14px;
    height: 14px;
  }
  button:focus-visible {
    outline: 2px solid var(--cyan);
    outline-offset: -2px;
  }
</style>
