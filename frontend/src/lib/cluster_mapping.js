export class ClusterMapping {
	/**
	 * @param {number} sourceIdx
	 * @param {number} clusterId
	 */
	constructor(sourceIdx, clusterId) {
		/** @type {number} */
		this.source_idx = sourceIdx
		/** @type {number} */
		this.cluster_id = clusterId
	}
}
