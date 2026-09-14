import { readFileSync } from 'node:fs';

import adapter from '@sveltejs/adapter-static';

// SvelteKit defaults kit.version.name to Date.now(), and derives the
// __sveltekit_<id> global it embeds in dist/index.html from it. That makes the
// built bundle different on every run, and dist/index.html is committed -- so
// any two branches that rebuild the web UI conflict in it, whether or not their
// source changes overlap.
//
// The package version is stable for a given source tree, so the bundle is
// reproducible, while still changing when the version is bumped -- which is
// what SvelteKit's client-side "app has been updated" check needs to keep
// working across releases.
const { version } = JSON.parse(
  readFileSync(new URL('./package.json', import.meta.url), 'utf8')
);

/** @type {import('@sveltejs/kit').Config} */
const config = {
  kit: {
    adapter: adapter({
      pages: 'dist',
      assets: 'dist',
      fallback: 'index.html',
      strict: true
    }),
    output: {
      bundleStrategy: 'inline'
    },
    router: {
      type: 'hash'
    },
    paths: {
      relative: true
    },
    version: {
      name: version
    }
  }
};

export default config;
