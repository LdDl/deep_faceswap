export class FileEntry {
	/**
	 * @param {string} name
	 * @param {boolean} is_dir
	 * @param {number} [size]
	 */
	constructor(name, is_dir, size) {
		/** @type {string} */
		this.name = name
		/** @type {boolean} */
		this.is_dir = is_dir
		/** @type {number|undefined} */
		this.size = size
	}

	/**
	 * @param {{name: string, is_dir: boolean, size?: number}} obj
	 * @returns {FileEntry}
	 */
	static fromJSON(obj) {
		return new FileEntry(obj.name, obj.is_dir, obj.size)
	}
}
