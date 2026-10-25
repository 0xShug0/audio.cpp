<script lang="ts">
  import type { ComponentType, SvelteComponent } from 'svelte';
  import { apps } from '$lib/apps/registry';
  import type { AppDefinition, AppProps } from '$lib/apps/types';
  export let active = true;
  let selected: AppDefinition | null = null;
  let component: ComponentType<SvelteComponent<AppProps>> | null = null;
  let instance: (SvelteComponent<AppProps> & { saveBeforeClose?: () => Promise<void> }) | null = null;
  let busy = false;
  let loading = false;
  let error = '';
  export function canClose() { return !busy && !loading; }
  async function open(app: AppDefinition) {
    if (!canClose()) return;
    selected = app; loading = true; error = '';
    try { component = (await app.load()).default; }
    catch (e) { error = `Could not open ${app.name}: ${String(e)}`; }
    finally { loading = false; }
  }
  async function back() {
    if (!canClose()) return;
    try { await instance?.saveBeforeClose?.(); selected = null; component = null; error = ''; }
    catch(e) { error = `Could not save the app before closing: ${String(e)}`; }
  }
</script>

<svelte:window on:beforeunload={event => { if (busy) { event.preventDefault(); event.returnValue = ''; } }}/>
<section class="apps-page">
  {#if selected}
    <header class="app-context">
      <button on:click={back} disabled={busy || loading}>← All apps</button><span>/</span><span>{selected.name}</span>
    </header>
    {#if loading}<p role="status">Opening app…</p>{/if}
    {#if error}<p role="alert">{error}</p>{/if}
    {#if component}<svelte:component this={component} bind:this={instance} {active} bind:busy/>{/if}
  {:else}
    <header class="app-heading"><div><p class="eyebrow">APPS</p><h1>Make something with your models</h1><p>Choose a workflow. Create and reuse speaker presets inside Text to speech.</p></div></header>
    <aside class="early-release" aria-label="Early release">
      <strong>Early release</strong>
      <p>Apps are still being developed. Workflows and saved project formats may change. Export audio you want to keep.</p>
    </aside>
    <div class="app-grid">
      {#each apps as app}
        <button class="app-card" on:click={() => open(app)}>
          <strong>{app.name}</strong><span>{app.description}</span><span class="app-open">Open app →</span>
        </button>
      {/each}
    </div>
  {/if}
</section>
<style>
  .app-context { display:flex; align-items:center; gap:12px; padding:6px 0; font-size:12px; color:var(--muted); }
  .apps-page { max-width:1440px; margin:0 auto; }
  .app-heading { display:flex; align-items:flex-start; gap:20px; margin:20px 0 32px; }
  h1 { margin:0 0 10px; font-size:28px; } p { color:var(--muted); margin:0; }
  .app-context button { flex-shrink:0; } .eyebrow { margin-bottom:12px; }
  .app-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:20px; }
  .early-release { margin-bottom:24px; padding:16px 20px; border:1px solid var(--cyan); border-radius:12px; background:var(--panel); }
  .early-release strong { color:var(--cyan); }
  .early-release p { margin-top:6px; line-height:1.6; }
  .app-card { display:flex; flex-direction:column; align-items:flex-start; text-align:left; gap:16px; padding:28px; min-height:240px; background:var(--panel); }
  .app-card strong { font-size:22px; } .app-card > span { color:var(--muted); line-height:1.6; }
  .app-card .app-open { color:var(--cyan); font-size:12px; } .app-open { margin-top:auto; }
  [role=alert] { color:var(--danger); }
  @media(max-width:600px) { .app-heading { flex-direction:column; } h1 { font-size:23px; } }
</style>
