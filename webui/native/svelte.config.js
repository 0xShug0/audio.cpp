import { readFileSync } from 'node:fs';

import adapter from '@sveltejs/adapter-static';

// SvelteKit defaults kit.version.name to Date.now() and derives the
// __sveltekit_<id> global it embeds in dist/index.html from it, so two builds
// of identical source differ. dist/index.html is committed -- it is embedded
// into the server binary at configure time, which is what lets the project
// build a working server without a JavaScript toolchain -- so any two branches
// that rebuild the web UI conflict in it whether or not their source changes
// overlap.
//
// Naming the version after the UI package makes the bundle a pure function of
// the source tree. Keep webui/native/package.json's version in step with the
// release it ships in; nothing derives it automatically, because audio.cpp's
// own version is tag-driven (AUDIOCPP_VERSION is passed at configure time and
// is not committed anywhere in the tree).
//
// Note that this is an identifier, not a live update channel: the app never
// imports SvelteKit's `updated` store, version.pollInterval is left at its
// default of 0, and the single-file bundle ships no _app/version.json for the
// client to poll. A stale version here costs nothing but a stale label.
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
