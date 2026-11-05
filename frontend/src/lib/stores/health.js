import { writable } from 'svelte/store'
import { request } from '../api.js'

export class HealthResponse {
	/**
	 * @param {string} status
	 * @param {boolean} [cuda]
	 * @param {boolean} [foregroundAvailable]
	 */
	constructor(status, cuda = false, foregroundAvailable = false) {
		/** @type {string} */
		this.status = status
		this.cuda = cuda
		this.foreground_available = foregroundAvailable
	}
}

/** @type {import('svelte/store').Writable<HealthResponse|null>} */
export const healthData = writable(null)

export const fetchHealth = () => {
	request('GET', '/health')
		.then((data) => {
			healthData.set(new HealthResponse(data.status, data.cuda, data.foreground_available))
		})
		.catch((error) => {
			healthData.set(new HealthResponse('unavailable'))
			console.error('Error on health check', error)
		})
}
