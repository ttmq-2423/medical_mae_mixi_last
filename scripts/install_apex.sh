#!/bin/sh
python -c 'import apex' && echo "[*] NVIDIA Apex already installed" || {
	APEX_REPO="${APEX_REPO:-./deps/apex}"
	echo "[*] Installing NVIDIA Apex"
        mkdir -p "$APEX_REPO"
        git clone --depth 1 https://github.com/NVIDIA/apex.git "$APEX_REPO"
	pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" "$APEX_REPO"
}
