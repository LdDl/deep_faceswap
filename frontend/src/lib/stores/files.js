import { writable } from 'svelte/store'
import { request } from '../api.js'
import { FileEntry } from '../file_entry.js'

/** @type {import('svelte/store').Writable<FileEntry[]>} */
export const filesData = writable([])

/** @type {import('svelte/store').Writable<boolean>} */
export const filesReady = writable(false)

/** @type {import('svelte/store').Writable<string>} */
export const filesError = writable('')

/**
 * @param {string} path
 */
export const fetchFiles = (path) => {
	filesReady.set(false)
	filesError.set('')
	request('GET', `/files?path=${encodeURIComponent(path)}`)
		.then((data) => {
			filesData.set(data.entries.map(FileEntry.fromJSON))
			filesReady.set(true)
		})
		.catch((error) => {
			console.error('Error on listing files', error)
			filesError.set(error.error_text || error.message)
			filesReady.set(true)
		})
}
