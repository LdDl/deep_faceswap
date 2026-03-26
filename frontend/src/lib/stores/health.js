import { writable } from 'svelte/store'
import { request } from '../api.js'

export class HealthResponse {
	/**
	 * @param {string} status
	 */
	constructor(status) {
		/** @type {string} */
		this.status = status
	}
}

/** @type {import('svelte/store').Writable<HealthResponse|null>} */
export const healthData = writable(null)

export const fetchHealth = () => {
	request('GET', '/health')
		.then((data) => {
			healthData.set(new HealthResponse(data.status))
		})
		.catch((error) => {
			console.error('Error on health check', error)
		})
}
