<script lang="ts">
  import type { CatalogEntry, InstallPackageChoice } from '$lib/types';
  import type { ModelInstallJob, ModelPackageSize } from '$lib/api';

  export let groups: { family: string; label: string; entries: CatalogEntry[] }[];
  export let packageSizes: Record<string, ModelPackageSize>;
  export let installJobs: Record<string, ModelInstallJob>;
  export let packageSizeState: string;
  export let modelsFolderInput: string;
  export let modelsFolder: string;
  export let modelsFolderHistory: string[] = [];
  export let defaultModelsFolder: string;
  export let applyingModelsFolder: boolean;
  export let modelsFolderIsDefault: boolean;
  export let openFolderBrowser: () => unknown;
  export let applyModelsFolder: (reset: boolean) => unknown;
  export let installPackage: (entry: CatalogEntry, choice: InstallPackageChoice, overwrite?: boolean) => unknown;
  export let removePackage: (entry: CatalogEntry, choice: InstallPackageChoice) => unknown;
  export let stopPackageDownload: (entry: CatalogEntry, job: ModelInstallJob) => unknown;
  export let cleanPartialDownload: (entry: CatalogEntry, job: ModelInstallJob) => unknown;
  export let openModel: (entry: CatalogEntry, choice?: InstallPackageChoice) => void;
  export let taskLabel: (entry: CatalogEntry) => string;
  export let formatBytes: (value: number) => string;
  export let installProgressLabel: (job: ModelInstallJob) => string;
  export let workflowForEntry: (entry: CatalogEntry) => string;
  export let workflows: { id: string; label: string }[];

  let search = '';
  let task = '';
  let installation = 'all';
  let choices: Record<string, string> = {};
  const active = (job?: ModelInstallJob) => !!job && ['queued', 'running', 'cancelling'].includes(job.state);
  function packageStatus(choice: InstallPackageChoice, sizes: Record<string, ModelPackageSize>, jobs: Record<string, ModelInstallJob>, sizeState: string) {
    if (sizes[choice.id]?.installed) return 'Installed';
    const job = jobs[choice.id];
    if (active(job)) return job.state === 'queued' ? 'Queued' : job.state === 'cancelling' ? 'Stopping' : 'Downloading';
    if (job && ['failed', 'cancelled'].includes(job.state)) return job.state === 'failed' ? 'Download failed' : 'Incomplete';
    if (!sizes[choice.id]) return sizeState === 'failed' ? 'Status unavailable' : 'Checking';
    return 'Not installed';
  }
  function selectedChoice(entry: CatalogEntry, selections: Record<string, string>, sizes: Record<string, ModelPackageSize>) {
    const packages = entry.install_packages || [];
    return packages.find((p) => p.id === selections[entry.id]) || packages.find((p) => sizes[p.id]?.installed) || packages[0];
  }
  function installedCount(entries: CatalogEntry[], sizes: Record<string, ModelPackageSize>) {
    return new Set(entries.flatMap((e) => (e.install_packages || []).filter((p) => sizes[p.id]?.installed).map((p) => p.id))).size;
  }
  $: visibleGroups = groups.map((group) => ({ ...group, entries: group.entries.filter((entry) => {
    const matches = `${group.label} ${entry.display_name} ${entry.family}`.toLowerCase().includes(search.trim().toLowerCase());
    const installed = installedCount([entry], packageSizes) > 0;
    const known = !!entry.install_packages?.length && entry.install_packages.every((p) => !!packageSizes[p.id]);
    return matches && (!task || workflowForEntry(entry) === task) && (installation === 'all' || (installation === 'installed' ? installed : known && !installed));
  }) })).filter((g) => g.entries.length);
  $: activeDownloads = Object.values(installJobs).filter(active);
</script>

<section class="library" aria-label="Model library">
  <header class="library-heading"><div><h1>Model Library</h1><span>{groups.reduce((n, g) => n + g.entries.length, 0)} models · {installedCount(groups.flatMap((g) => g.entries), packageSizes)} packages installed</span></div>
    {#if activeDownloads.length}<span class="download-count">{activeDownloads.length} active downloads</span>{/if}
  </header>
  <details class="folder-settings"><summary>Models folder <span>{modelsFolder || defaultModelsFolder}</span></summary>
    <div class="folder-editor"><input aria-label="Models folder" bind:value={modelsFolderInput} placeholder={defaultModelsFolder} />
      <button disabled={applyingModelsFolder} on:click={() => openFolderBrowser()}>Browse</button>
      <button disabled={applyingModelsFolder || !modelsFolderInput.trim() || modelsFolderInput.trim() === modelsFolder} on:click={() => applyModelsFolder(false)}>Apply</button>
      <button disabled={applyingModelsFolder || modelsFolderIsDefault} on:click={() => applyModelsFolder(true)}>Use default</button></div>
    <div class="folder-history">
      <label for="models-folder-history">Recent folders</label>
      <select id="models-folder-history" disabled={applyingModelsFolder || !modelsFolderHistory.length}
        value={modelsFolderHistory.includes(modelsFolderInput) ? modelsFolderInput : ''}
        on:change={(event) => modelsFolderInput = event.currentTarget.value}>
        <option value="" disabled>Choose a recent folder</option>
        {#each modelsFolderHistory as path}<option value={path}>{path}</option>{/each}
      </select>
    </div>
  </details>
  <div class="filters">
    <input type="search" aria-label="Search models" placeholder="Search models or families" bind:value={search} />
    <select aria-label="Task" bind:value={task}><option value="">All tasks</option>{#each workflows as w}<option value={w.id}>{w.label}</option>{/each}</select>
    <select aria-label="Installation status" bind:value={installation}><option value="all">All models</option><option value="installed">Installed</option><option value="missing">Not installed</option></select>
  </div>
  <div class="result-count">{visibleGroups.reduce((n, g) => n + g.entries.length, 0)} models · A–Z</div>
  {#if activeDownloads.length}
    <details class="downloads" open><summary>Downloads</summary>{#each activeDownloads as job}
      {@const entry = groups.flatMap((g) => g.entries).find((e) => e.install_packages?.some((p) => p.id === job.id))}
      <div class="download-row"><span>{entry?.display_name || job.id}</span><span>{installProgressLabel(job)}</span>{#if entry}<button disabled={job.state === 'cancelling'} on:click={() => stopPackageDownload(entry, job)}>Stop</button>{/if}</div>
    {/each}</details>
  {/if}
  {#each visibleGroups as group (group.family)}
    <section class="family">
      {#if group.entries.length > 1}<header><h2>{group.label}</h2><span class:installed={installedCount(group.entries, packageSizes) > 0}>{installedCount(group.entries, packageSizes) ? `${installedCount(group.entries, packageSizes)} package${installedCount(group.entries, packageSizes) === 1 ? '' : 's'} installed` : group.entries.every((e) => !e.install_packages?.length) ? 'Manual setup' : group.entries.some((e) => e.install_packages?.some((p) => !packageSizes[p.id])) ? 'Status pending' : 'Not installed'}</span></header>{/if}
      {#each group.entries as entry (entry.id)}
        {@const packageCount = new Set((entry.install_packages || []).map((p) => p.id)).size}
        {@const choice = selectedChoice(entry, choices, packageSizes)}
        {@const job = choice ? installJobs[choice.id] : undefined}
        {@const installed = choice && packageSizes[choice.id]?.installed}
        {@const busy = group.entries.some((e) => e.install_packages?.some((p) => active(installJobs[p.id])))}
        <div class="model-row">
          <div class="identity"><strong>{entry.display_name}</strong><span>{taskLabel(entry)}</span>{#if packageCount && entry.install_packages?.every((p) => !!packageSizes[p.id])}<span>{installedCount([entry], packageSizes)} of {packageCount} packages installed</span>{/if}</div>
          {#if choice}
            <div class="package"><label for={`package-${entry.id}`}>Package ({packageCount})</label><select id={`package-${entry.id}`} value={choice.id} on:change={(event) => choices = { ...choices, [entry.id]: event.currentTarget.value }}>
              {#each entry.install_packages || [] as p}<option value={p.id}>{p.label} · {packageStatus(p, packageSizes, installJobs, packageSizeState)}</option>{/each}
            </select><span>{packageSizes[choice.id]?.size_bytes ? `${formatBytes(packageSizes[choice.id].size_bytes!)} download` : 'Download size unavailable'}</span></div>
            <div class="state" class:installed><strong>{installed ? '✓ Installed' : packageStatus(choice, packageSizes, installJobs, packageSizeState)}</strong>{#if installed && packageSizes[choice.id]?.version_state === 'update_available'}<span>Update available</span>{/if}</div>
            <div class="actions"><button class="primary" disabled={!installed && (active(job) || !packageSizes[choice.id])} on:click={() => installed ? openModel(entry, choice) : installPackage(entry, choice)}>{installed ? 'Open in Studio' : 'Download'}</button>
              {#if installed}<details class="manage"><summary aria-label={`Manage ${entry.display_name}`}>Manage</summary><div>{#if packageSizes[choice.id]?.version_state === 'update_available'}<button disabled={busy} on:click={() => installPackage(entry, choice, true)}>Update</button>{/if}<button disabled={busy} on:click={() => removePackage(entry, choice)}>Remove</button></div></details>{/if}
            </div>
            {#if job && !['complete', 'cleaned'].includes(job.state)}<div class="job" class:failed={job.state === 'failed'}><span>{installProgressLabel(job)}</span><progress max="100" value={job.progress_percent >= 0 ? job.progress_percent : undefined}></progress><p>{job.message}</p>{#if active(job)}<button disabled={job.state === 'cancelling'} on:click={() => stopPackageDownload(entry, job)}>Stop download</button>{:else}<button on:click={() => cleanPartialDownload(entry, job)}>Clean partial download</button>{/if}</div>{/if}
          {:else}<span class="state">Manual setup</span><div class="actions">{#if entry.weights_url}<a href={entry.weights_url} target="_blank" rel="noreferrer">View weights ↗</a>{/if}<button on:click={() => openModel(entry)}>Open in Studio</button></div>{/if}
        </div>
      {/each}
    </section>
  {:else}<div class="empty">No models match these filters.<button on:click={() => { search = ''; task = ''; installation = 'all'; }}>Clear filters</button></div>{/each}
</section>

<style>
  .library { max-width: 1400px; margin: 0 auto; letter-spacing: 0; }
  .library-heading, .family > header { display: flex; align-items: center; justify-content: space-between; gap: 16px; }
  h1 { font-size: 26px; margin: 0 0 8px; } h2 { font-size: 16px; margin: 0; }
  .library-heading { padding: 20px 0; } .library-heading span, .result-count, .identity span, .package span, .package label, .state span { color: var(--muted); font-size: 13px; }
  .folder-settings { border-bottom: 1px solid var(--line); padding: 12px 0; }
  summary { cursor: pointer; font-size: 13px; } summary span { color: var(--muted); margin-left: 12px; overflow-wrap: anywhere; }
  .folder-editor { display: flex; gap: 8px; padding: 12px 0; clear: both; } .folder-editor input { flex: 1; min-width: 0; }
  .folder-history { display: grid; gap: 6px; padding-bottom: 12px; min-width: 0; }
  .folder-history label { color: var(--muted); font-size: 13px; }
  .folder-history select { width: 100%; min-width: 0; }
  .filters { display: grid; grid-template-columns: minmax(180px, 1fr) 220px 180px; gap: 12px; padding: 24px 0 12px; }
  input, select, button { font-size: 13px; min-height: 38px; border-radius: 6px; letter-spacing: 0; }
  .result-count { padding-bottom: 16px; } .family { margin: 0; }
  .family > header { margin-top: 12px; padding: 14px 12px; background: var(--panel); } .family > header > span { font-size: 12px; color: var(--muted); }
  .model-row { display: grid; grid-template-columns: minmax(170px, 1fr) minmax(160px, 240px) 150px 180px; align-items: center; gap: 20px; padding: 18px 12px; border-bottom: 1px solid var(--line); }
  .identity, .package, .state { display: grid; gap: 6px; min-width: 0; } .identity strong { font-size: 14px; overflow-wrap: anywhere; } .package select { width: 100%; }
  .state { font-size: 13px; } .installed, .family > header > .installed { color: var(--green, #a6e3a1); }
  .actions { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; } .primary { width: 100%; }
  .manage > div { display: flex; gap: 8px; padding-top: 8px; } .job { grid-column: 1/-1; font-size: 13px; overflow-wrap: anywhere; }
  progress { display: block; width: 100%; height: 6px; margin-top: 10px; accent-color: var(--cyan); } .failed { color: var(--danger); }
  .downloads { padding: 12px; border: 1px solid var(--line); margin-bottom: 20px; } .download-row { display: flex; gap: 16px; align-items: center; padding-top: 10px; font-size: 13px; } .download-row span:first-child { flex: 1; }
  .empty { padding: 48px 0; display: grid; justify-items: center; gap: 16px; }
  @media (max-width: 1050px) { .model-row { grid-template-columns: minmax(140px, 1fr) minmax(150px, 1fr); } .actions { justify-self: end; } .filters { grid-template-columns: 1fr 1fr; } .filters input { grid-column: 1/-1; } }
  @media (max-width: 600px) { .model-row { grid-template-columns: 1fr; gap: 12px; } .actions { justify-self: stretch; } .family > header, .library-heading { align-items: start; flex-direction: column; } .folder-editor { flex-wrap: wrap; } .folder-editor input { flex-basis: 100%; } .filters { grid-template-columns: minmax(0, 1fr); } .download-row { flex-wrap: wrap; } }
</style>
