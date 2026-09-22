ARG KEELSON_IMAGE=ghcr.io/rise-maritime/keelson:0.6.0-pre.15
FROM ${KEELSON_IMAGE}

RUN python3 -m pip install --no-cache-dir "numpy>=1.26,<3" "jsonschema>=4.18,<5"

COPY src/vessel_extraction_core.py /usr/local/lib/vessel_extraction_core.py
COPY src/pointcloud_vessel_extraction2keelson.py /usr/local/bin/pointcloud-vessel-extraction2keelson
RUN chmod +x /usr/local/bin/pointcloud-vessel-extraction2keelson

ENTRYPOINT ["pointcloud-vessel-extraction2keelson"]
