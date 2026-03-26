.PHONY: all frontend api cli clean

FRONTEND_DIR = frontend
FRONTEND_BUILD = $(FRONTEND_DIR)/build
API_BIN = target/release/deep-faceswap-api
CLI_BIN = target/release/deep-faceswap-cli

# Build everything: frontend first, then Rust binaries
all: frontend api cli

# Build SvelteKit frontend (static adapter)
frontend:
	@echo "Building frontend..."
	cd $(FRONTEND_DIR) && npm install && npm run build
	@echo "Frontend built: $(FRONTEND_BUILD)/"

# Build API server (release, with CUDA)
api:
	@echo "Building API server..."
	cargo build --release --features cuda -p deep_faceswap_api
	@echo "API built: $(API_BIN)"

# Build CLI tool (release, with CUDA)
cli:
	@echo "Building CLI..."
	cargo build --release --features cuda -p deep_faceswap_cli
	@echo "CLI built: $(CLI_BIN)"

# Run API server with embedded frontend
run: all
	$(API_BIN) \
		--ui-dir $(FRONTEND_BUILD) \
		--detector models/buffalo_l/det_10g.onnx \
		--recognizer models/buffalo_l/w600k_r50.onnx \
		--swapper models/inswapper_128.onnx \
		--enhancer models/GFPGANv1.4.onnx \
		--landmark-model models/buffalo_l/2d106det.onnx \
		--port 36000

clean:
	cargo clean
	rm -rf $(FRONTEND_BUILD)
	rm -rf $(FRONTEND_DIR)/node_modules/.vite
