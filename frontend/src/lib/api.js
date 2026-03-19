import Connection from './connection.js'
import { ApiError } from './api_error.js'

/* global __API_BASE_URL__ */
const conn = new Connection(__API_BASE_URL__)

/**
 * @param {string} method
 * @param {string} path
 * @param {object} [body]
 * @returns {Promise<any>}
 */
export async function request(method, path, body) {
	const url = conn.getAddress(path)
	/** @type {RequestInit} */
	const opts = { method }
	if (body !== undefined) {
		opts.headers = { 'Content-Type': 'application/json' }
		opts.body = JSON.stringify(body)
	}
	const res = await fetch(url, opts)
	const data = await res.json()
	if (!res.ok) {
		throw new ApiError(res.status, data.error_text || `HTTP ${res.status}`)
	}
	return data
}

/**
 * @param {string} serverPath
 * @returns {string}
 */
export function cropUrl(serverPath) {
	const relative = serverPath.replace(/^\/api\//, '/')
	return conn.getAddress(relative)
}

/**
 * @param {string} absolutePath
 * @returns {string}
 */
export function fileUrl(absolutePath) {
	return conn.getAddress(`/file?path=${encodeURIComponent(absolutePath)}`)
}
