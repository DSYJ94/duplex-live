# Build from repository root. No model weights, datasets or credentials in image.
FROM nvcr.io/nvidia/pytorch:25.10-py3@sha256:42263b2424fc237b34c4fc4a91c30d603c57eed36e37d31ff6d9a4f1f801edee
RUN python -m venv --system-site-packages /opt/duplex-env
ENV PATH="/opt/duplex-env/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_OFFLINE=1 \
    CUDA_VISIBLE_DEVICES=0 \
    OMP_NUM_THREADS=8
COPY requirements.txt /opt/duplex-requirements.txt
RUN pip install --no-cache-dir -r /opt/duplex-requirements.txt \
 && python -c 'import torch; assert str(torch.__version__) == "2.9.0a0+145a3a7bda.nv25.10", torch.__version__'
# Source and artifact trees are mounted at their original provenance paths.
WORKDIR /projects/u6si/duplex-live
ENTRYPOINT ["python", "-u"]
