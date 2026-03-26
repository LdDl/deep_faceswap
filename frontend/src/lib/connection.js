/**
 * @param {string} baseURL
 */
export default function Connection(baseURL) {
	/** @type {string} */
	this.baseURL = baseURL.replace(/\/+$/, '')

	/**
	 * @param {string} url
	 * @returns {boolean}
	 */
	this.isUrlAbsolute = function (url) {
		return (url.indexOf('://') > 0 || url.indexOf('//') === 0)
	}

	/**
	 * @param {string} path
	 * @returns {string}
	 */
	this.getAddress = function (path = '') {
		if (this.isUrlAbsolute(path)) {
			return path
		}

		if (!this.isUrlAbsolute(this.baseURL)) {
			const origin = `${window.location.protocol}//${window.location.host}`
			const cleanBase = this.baseURL.replace(/^\/+/, '').replace(/\/+$/, '')
			const cleanPath = path.replace(/^\/+/, '')
			return `${origin}/${cleanBase}/${cleanPath}`
		}

		return `${this.baseURL}/${path.replace(/^\//, '')}`
	}

	/**
	 * @param {string} path
	 * @returns {string}
	 */
	this.getWsAddress = function (path = '') {
		const httpUrl = this.getAddress(path)
		return httpUrl.replace(/^http/, 'ws')
	}
}
