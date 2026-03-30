import { request } from '../api.js'

export class JobProgress {
	/**
	 * @param {string} stage
	 * @param {number} current
	 * @param {number} total
	 */
	constructor(stage, current, total) {
		/** @type {string} */
		this.stage = stage
		/** @type {number} */
		this.current = current
		/** @type {number} */
		this.total = total
	}
}

export class JobResult {
	/**
	 * @param {string} [output_path]
	 * @param {any} [data]
	 */
	constructor(output_path, data) {
		/** @type {string|undefined} */
		this.output_path = output_path
		/** @type {any} */
		this.data = data
	}
}

export class JobState {
	/**
	 * @param {string} job_id
	 * @param {string} status
	 * @param {JobProgress} progress
	 * @param {JobResult} [result]
	 * @param {string} [error]
	 */
	constructor(job_id, status, progress, result, error) {
		/** @type {string} */
		this.job_id = job_id
		/** @type {string} */
		this.status = status
		/** @type {JobProgress} */
		this.progress = progress
		/** @type {JobResult|undefined} */
		this.result = result
		/** @type {string|undefined} */
		this.error = error
	}
}

/**
 * @param {string} jobId
 * @returns {Promise<JobState>}
 */
export const getJobStatus = async (jobId) => {
	const data = await request('GET', `/jobs/${encodeURIComponent(jobId)}`)
	return new JobState(
		data.id,
		data.status,
		new JobProgress(data.progress.stage, data.progress.current, data.progress.total),
		data.result ? new JobResult(data.result.output_path, data.result.data) : undefined,
		data.error
	)
}
