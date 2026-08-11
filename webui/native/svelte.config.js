import adapter from '@sveltejs/adapter-static';

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
    paths: {
      relative: true
    }
  }
};

export default config;
