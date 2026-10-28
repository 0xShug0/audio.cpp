import { catalog, parameterCatalog, requirePackageLoadTarget, withPackageLoadTarget } from '$lib/catalog';
import { health, models, modelPackageSizes } from '$lib/api';
import type { CatalogEntry, InstallPackageChoice } from '$lib/types';
import type { ModelSettings } from './types';

// Installed model choices become configuration snapshots owned by the app.
export async function loadModelChoices(): Promise<ModelSettings[]> {
  const [server, registered] = await Promise.all([health(), models()]);
  const choices: ModelSettings[] = [];
  const configured = registered.filter(model => !server.ui_management || model.configured);
  for (const model of configured) {
    const entry = catalog.find(entry => entry.id === model.id) || catalog.find(entry => entry.family === model.family && entry.task === model.task);
    choices.push(modelChoice({ ...entry, ...model, display_name: entry?.display_name || model.id, load_options: undefined } as CatalogEntry,
      undefined, model.path, model.session_options || {}));
  }
  if (server.ui_management) {
    const inventory = await modelPackageSizes();
    const rows = new Map(inventory.data.map(row => [row.id, row]));
    for (const entry of catalog.filter(e => !configured.some(model => model.id === e.id))) {
      const installed = (entry.install_packages || []).map(p => withPackageLoadTarget(p, rows.get(p.id)))
        .filter(p => rows.get(p.id)?.installed && p.load_target);
      for (const choice of installed) {
        choices.push(modelChoice(entry, choice, requirePackageLoadTarget(choice).model_path));
      }
      if (!installed.length) {
        const resident = registered.find(m => m.id === entry.id && m.loaded);
        if (resident) choices.push(modelChoice(entry, undefined, resident.path, resident.session_options));
      }
    }
  }
  return choices;
}
function modelChoice(entry: CatalogEntry, choice: InstallPackageChoice | undefined, path: string,
  sessionOptions = { ...entry.session_options, ...choice?.session_options }): ModelSettings {
  const parameters = parameterCatalog[entry.id] || parameterCatalog[entry.family] || [];
  const options = Object.fromEntries(parameters.filter(p => p.scope !== 'session' && p.default !== undefined && p.default !== '' &&
    (entry.request_options === undefined || entry.request_options.includes(p.name))).map(p => [p.name, p.default]));
  return { version: 1, id: `model:${entry.id}:${choice?.id || 'configured'}`,
    name: `${entry.display_name}${choice ? ` · ${choice.label}` : ''}`, createdAt: 0,
    model: { id: entry.id, display_name: entry.display_name, family: choice?.load_target?.family ?? entry.family, task: entry.task,
      mode: entry.mode, path, package_id: choice?.id,
      load_options: { ...choice?.load_options, ...entry.load_options }, session_options: sessionOptions },
    request: { seed: -1, ...(entry.default_voice ? { voice: entry.default_voice } : {}), options: { ...entry.default_options, ...options } },
    chunking: { enabled: true, characters: 400 } };
}
