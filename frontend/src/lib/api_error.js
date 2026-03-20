export class ApiError extends Error {
	/**
	 * @param {number} status
	 * @param {string} errorText
	 */
	constructor(status, errorText) {
		super(errorText)
		/** @type {number} */
		this.status = status
		/** @type {string} */
		this.error_text = errorText
	}
}
