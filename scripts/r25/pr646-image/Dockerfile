FROM voipmonitor/vllm@sha256:89376e9aa49442a90754662ca1bb281bffbeca29bb7393e6e8281506e5ac4804

COPY --chmod=755 serve-glm53-flash.sh /usr/local/bin/serve-glm53-flash.sh
COPY vllm/platforms/interface.py /opt/glm53-flash/vllm/vllm/platforms/interface.py
COPY vllm/v1/core/kv_cache_coordinator.py /opt/glm53-flash/vllm/vllm/v1/core/kv_cache_coordinator.py
COPY vllm/v1/core/single_type_kv_cache_manager.py /opt/glm53-flash/vllm/vllm/v1/core/single_type_kv_cache_manager.py
COPY tests/v1/core/prefix_cache/test_partial_prefix_cache_hits.py /opt/glm53-flash/vllm/tests/v1/core/prefix_cache/test_partial_prefix_cache_hits.py
COPY pr646.lock /opt/glm53-flash/pr646.lock

LABEL local-inference-lab.test.base="r25@sha256:89376e9aa49442a90754662ca1bb281bffbeca29bb7393e6e8281506e5ac4804" \
      local-inference-lab.test.patch="local-inference-lab/vllm#646" \
      local-inference-lab.test.scope="python-overlay-no-push"
