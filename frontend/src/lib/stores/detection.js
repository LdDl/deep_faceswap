import { request } from '../api.js'
import { FaceInfo } from '../face_info.js'

export class DetectionResponse {
	/**
	 * @param {string} session_id
	 * @param {FaceInfo[]} source_faces
	 * @param {FaceInfo[]} target_faces
	 */
	constructor(session_id, source_faces, target_faces) {
		/** @type {string} */
		this.session_id = session_id
		/** @type {FaceInfo[]} */
		this.source_faces = source_faces
		/** @type {FaceInfo[]} */
		this.target_faces = target_faces
	}
}

/**
 * @param {string[]} sourcePaths
 * @param {string} targetPath
 * @returns {Promise<DetectionResponse>}
 */
export const detectFaces = async (sourcePaths, targetPath) => {
	const data = await request('POST', '/detect', {
		source_paths: sourcePaths,
		target_path: targetPath
	})
	return new DetectionResponse(
		data.session_id,
		data.source_faces.map(FaceInfo.fromJSON),
		data.target_faces.map(FaceInfo.fromJSON)
	)
}
