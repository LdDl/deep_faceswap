import { request } from '../api.js'
import { FaceMapping } from '../face_mapping.js'

export class SwapImageResponse {
	/**
	 * @param {string} output_path
	 * @param {number} elapsed_s
	 * @param {number} faces_swapped
	 */
	constructor(output_path, elapsed_s, faces_swapped) {
		/** @type {string} */
		this.output_path = output_path
		/** @type {number} */
		this.elapsed_s = elapsed_s
		/** @type {number} */
		this.faces_swapped = faces_swapped
	}
}

/**
 * @param {string[]} sourcePaths
 * @param {string} targetPath
 * @param {string} outputPath
 * @param {FaceMapping[]} mappings
 * @param {boolean} enhance
 * @param {boolean} mouthMask
 * @returns {Promise<SwapImageResponse>}
 */
export const swapImage = async (sourcePaths, targetPath, outputPath, mappings, enhance, mouthMask) => {
	const data = await request('POST', '/swap/image', {
		source_paths: sourcePaths,
		target_path: targetPath,
		output_path: outputPath,
		mappings: mappings,
		enhance: enhance,
		mouth_mask: mouthMask
	})
	return new SwapImageResponse(data.output_path, data.elapsed_s, data.faces_swapped)
}
