import { sveltekit } from '@sveltejs/kit/vite';
import tailwindcss from '@tailwindcss/vite';
import { defineConfig } from 'vite';

// Default API URL for local development
const DEFAULT_URL = 'http://localhost:36000/api'

// In production, override with API_BASE_URL environment variable at build time:
//
//   npm run build                                         => /api (default, same origin)
//   API_BASE_URL=/api npm run build                       => /api (same origin, relative)
//   API_BASE_URL=/potato npm run build                    => /potato (reverse proxy path)
//   API_BASE_URL=/my/deep/proxy/path npm run build        => /my/deep/proxy/path (nested proxy)
//   API_BASE_URL=https://api.example.com/v1 npm run build => https://api.example.com/v1 (absolute)
//
// Relative paths are resolved against window.location.origin at runtime.
// For example, /potato on https://example.com becomes https://example.com/potato/health when fetching /potato/health.
//
// In dev mode DEFAULT_URL is always used, proxy forwards /api to the backend.

const isDev = process.env.NODE_ENV !== 'production';
const effectiveUrl = isDev ? DEFAULT_URL : (process.env.API_BASE_URL || '/api');

export default defineConfig({
	define: {
		'__API_BASE_URL__': JSON.stringify(effectiveUrl)
	},
	plugins: [tailwindcss(), sveltekit()],
	server: {
		proxy: {
			'/api': DEFAULT_URL.replace(/\/api\/?$/, '')
		}
	}
});
