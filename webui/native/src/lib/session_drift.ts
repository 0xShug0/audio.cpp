import type { ParamSpec, StringMap } from './types';

// Compare a resident model's reported session_options against the UI's expected
// session options for package-managed and session-scoped parameter keys.
//
// When a server omits a key (for example, a server-configured model relying on
// engine defaults), an expected value equal to the parameter's declared default
// is treated as matching so unchanged defaults do not trigger a false reload prompt.
export function sessionOptionsMatch(
  actual: StringMap | undefined,
  expected: StringMap,
  sessionSpecs: readonly ParamSpec[],
  extraKeys: readonly string[] = []
): boolean {
  const resident = actual || {};
  const specByKey = new Map<string, ParamSpec>();
  const keys = new Set<string>(extraKeys);
  for (const spec of sessionSpecs) {
    if (spec.scope !== 'session') continue;
    const key = spec.session_option || spec.name;
    specByKey.set(key, spec);
    keys.add(key);
  }
  for (const key of keys) {
    const actualValue = resident[key];
    const expectedValue = expected[key];
    if (actualValue === expectedValue) continue;
    if (actualValue === undefined) {
      if (expectedValue === undefined || expectedValue === '') continue;
      const spec = specByKey.get(key);
      if (spec && spec.default !== undefined && String(spec.default) === expectedValue) continue;
      return false;
    }
    if (expectedValue === undefined || expectedValue === '') {
      if (actualValue === '') continue;
      return false;
    }
    return false;
  }
  return true;
}
