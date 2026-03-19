export class BBox {
	/**
	 * @param {number} x1
	 * @param {number} y1
	 * @param {number} x2
	 * @param {number} y2
	 * @param {number} score
	 */
	constructor(x1, y1, x2, y2, score) {
		/** @type {number} */
		this.x1 = x1
		/** @type {number} */
		this.y1 = y1
		/** @type {number} */
		this.x2 = x2
		/** @type {number} */
		this.y2 = y2
		/** @type {number} */
		this.score = score
	}
}

export class FaceInfo {
	/**
	 * @param {number} index
	 * @param {BBox} bbox
	 * @param {number} det_score
	 * @param {string} crop_url
	 * @param {number} [source_image_index]
	 * @param {string} [source_filename]
	 */
	constructor(index, bbox, det_score, crop_url, source_image_index, source_filename) {
		/** @type {number} */
		this.index = index
		/** @type {BBox} */
		this.bbox = bbox
		/** @type {number} */
		this.det_score = det_score
		/** @type {string} */
		this.crop_url = crop_url
		/** @type {number|undefined} */
		this.source_image_index = source_image_index
		/** @type {string|undefined} */
		this.source_filename = source_filename
	}

	/**
	 * @param {{index: number, bbox: {x1: number, y1: number, x2: number, y2: number, score: number}, det_score: number, crop_url: string, source_image_index?: number, source_filename?: string}} obj
	 * @returns {FaceInfo}
	 */
	static fromJSON(obj) {
		return new FaceInfo(
			obj.index,
			new BBox(obj.bbox.x1, obj.bbox.y1, obj.bbox.x2, obj.bbox.y2, obj.bbox.score),
			obj.det_score,
			obj.crop_url,
			obj.source_image_index,
			obj.source_filename
		)
	}
}
