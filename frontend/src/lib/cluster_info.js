export class ClusterInfo {
	/**
	 * @param {number} cluster_id
	 * @param {number} frame_count
	 * @param {string} crop_url
	 */
	constructor(cluster_id, frame_count, crop_url) {
		/** @type {number} */
		this.cluster_id = cluster_id
		/** @type {number} */
		this.frame_count = frame_count
		/** @type {string} */
		this.crop_url = crop_url
	}

	/**
	 * @param {{cluster_id: number, frame_count: number, crop_url: string}} obj
	 * @returns {ClusterInfo}
	 */
	static fromJSON(obj) {
		return new ClusterInfo(obj.cluster_id, obj.frame_count, obj.crop_url)
	}
}
