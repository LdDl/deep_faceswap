import { request } from '../api.js'
import { ClusterMapping } from '../cluster_mapping.js'
import { FaceInfo } from '../face_info.js'
import { ClusterInfo } from '../cluster_info.js'

export class VideoAnalyzeResponse {
	/**
	 * @param {string} session_id
	 * @param {FaceInfo[]} source_faces
	 * @param {ClusterInfo[]} clusters
	 * @param {number} total_frames
	 * @param {number} elapsed_s
	 */
	constructor(session_id, source_faces, clusters, total_frames, elapsed_s) {
		/** @type {string} */
		this.session_id = session_id
		/** @type {FaceInfo[]} */
		this.source_faces = source_faces
		/** @type {ClusterInfo[]} */
		this.clusters = clusters
		/** @type {number} */
		this.total_frames = total_frames
		/** @type {number} */
		this.elapsed_s = elapsed_s
	}
}

export class VideoSwapResponse {
	/**
	 * @param {string} job_id
	 * @param {string} status
	 */
	constructor(job_id, status) {
		/** @type {string} */
		this.job_id = job_id
		/** @type {string} */
		this.status = status
	}
}

/**
 * @param {string[]} sourcePaths
 * @param {string} targetVideoPath
 * @param {string} [tmpDir]
 * @returns {Promise<VideoAnalyzeResponse>}
 */
export const analyzeVideo = async (sourcePaths, targetVideoPath, tmpDir) => {
	const body = {
		source_paths: sourcePaths,
		target_video_path: targetVideoPath
	}
	if (tmpDir) body.tmp_dir = tmpDir
	const data = await request('POST', '/video/analyze', body)
	return new VideoAnalyzeResponse(
		data.session_id,
		data.source_faces.map(FaceInfo.fromJSON),
		data.clusters.map(ClusterInfo.fromJSON),
		data.total_frames,
		data.elapsed_s
	)
}

/**
 * @param {string} sessionId
 * @param {string[]} sourcePaths
 * @param {string} targetVideoPath
 * @param {string} outputPath
 * @param {ClusterMapping[]} clusterMappings
 * @param {boolean} enhance
 * @param {boolean} mouthMask
 * @param {string} [tmpDir]
 * @returns {Promise<VideoSwapResponse>}
 */
export const swapVideo = async (sessionId, sourcePaths, targetVideoPath, outputPath, clusterMappings, enhance, mouthMask, tmpDir) => {
	const body = {
		session_id: sessionId,
		source_paths: sourcePaths,
		target_video_path: targetVideoPath,
		output_path: outputPath,
		cluster_mappings: clusterMappings,
		enhance: enhance,
		mouth_mask: mouthMask
	}
	if (tmpDir) body.tmp_dir = tmpDir
	const data = await request('POST', '/swap/video', body)
	return new VideoSwapResponse(data.job_id, data.status)
}
