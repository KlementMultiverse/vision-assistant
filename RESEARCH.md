# Vision Assistant Research: Best Practices 2026

> **Purpose:** Evidence-based recommendations for building a production-grade home AI vision system.
>
> **Last Updated:** 2026-02-07
> **Researcher:** Claude (RESEARCHER agent)

---

## Executive Summary

This document provides research-backed recommendations for the 7 key components of the Vision Assistant Home AI system. All recommendations are grounded in 2026 best practices with cited sources.

---

## 1. Multi-Camera State Management

### Problem Statement
Managing state across multiple cameras requires handling synchronization, failures, and consistent views of the world.

### Best Practices

#### 1.1 Centralized State with Home Assistant Pattern

**Recommendation:** Use a centralized hub (like Home Assistant) that maintains camera states.

| Pattern | Description | Use Case |
|---------|-------------|----------|
| Hub-Network-Device | Hub gives commands, network communicates (Wi-Fi/Thread/MQTT), devices execute | Home automation standard |
| MQTT Status | Each camera publishes status to MQTT topics | Real-time state sync |
| Entity Registry | Central registry of all devices with current state | Single source of truth |

**Implementation:**
```python
# Camera state model
class CameraState:
    camera_id: str
    status: Literal["online", "offline", "error"]
    last_frame_time: datetime
    error_count: int
    config: CameraConfig
```

#### 1.2 Camera Failure Detection

**Pattern:** Last Will and Testament (LWT) with MQTT

When a camera disconnects unexpectedly, the broker publishes the LWT, allowing the system to detect the outage immediately.

```python
# MQTT LWT setup
client.will_set(
    topic=f"cameras/{camera_id}/status",
    payload="offline",
    retain=True
)
```

**Stats Monitoring:** Publish health metrics at regular intervals (default: 60 seconds) to detect gradual degradation.

#### 1.3 Automatic Reconnection Strategy

**Recommendation:** Exponential backoff with jitter

```python
async def reconnect_camera(camera_id: str, max_attempts: int = 5):
    for attempt in range(max_attempts):
        delay = min(2 ** attempt + random.uniform(0, 1), 60)
        await asyncio.sleep(delay)
        if await try_connect(camera_id):
            return True
    return False
```

### Sources
- [Home Assistant](https://www.home-assistant.io/)
- [Frigate NVR MQTT Integration](https://docs.frigate.video/integrations/home-assistant/)
- [IoT Security Best Practices](https://www.connectwise.com/blog/how-to-secure-iot-devices)

---

## 2. Real-time Person Tracking & Re-identification

### Problem Statement
Tracking the same person across multiple cameras with different viewpoints, lighting, and occlusions.

### Best Practices

#### 2.1 Two-Aspect Approach

**Key Insight:** Treat multi-camera Re-ID as two separate problems:

1. **Temporal Tracking:** Tracking through time within a single camera view
2. **Spatial Re-ID:** Matching appearance across different cameras

#### 2.2 Recommended Pipeline

| Stage | Model/Approach | Purpose |
|-------|----------------|---------|
| Detection | YOLOv8n | Lightweight person detection |
| Tracking | DeepSORT (MARS dataset) | Multi-object tracking within camera |
| Re-ID | OSNet_x1_0 or ResNet50 | Cross-camera appearance matching |
| Trajectory | B-spline + Homography | Smooth trajectory comparison |

**Performance Benchmarks:**
- OSNet_x1_0: 98.4% mAP on Market1501
- ResNet50: 96.1% mAP on Market1501

#### 2.3 Person Lifecycle State Machine

**Key States:**

```
┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│ Just Arrived│ ──▶  │    Home     │ ──▶  │   Away      │
└─────────────┘      └─────────────┘      └─────────────┘
       │                    │                    │
       ▼                    ▼                    ▼
┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│ Short Trip  │      │  Idle/Sleep │      │Extended Away│
└─────────────┘      └─────────────┘      └─────────────┘
```

**State Transitions:**
- `Just Arrived` → `Home`: After 5 minutes of presence
- `Home` → `Away`: No detection for 15 minutes
- `Away` → `Extended Away`: Gone > 24 hours

#### 2.4 Embedding Management

**Critical:** Expire embeddings after a certain time to prevent:
- Uncontrolled memory usage
- Stale appearance data (clothing changes)

```python
# Embedding with TTL
class PersonEmbedding:
    embedding: np.ndarray
    created_at: datetime
    ttl_hours: int = 4  # Expire after 4 hours
```

#### 2.5 Edge Computing Benefits

- **Privacy:** No raw video sent to cloud
- **Latency:** Critical for real-time alerts
- **Bandwidth:** Only processed/encoded data transmitted

### Sources
- [Hailo Multi-Camera Re-ID](https://hailo.ai/blog/multi-camera-multi-person-re-identification/)
- [CVPR 2024 Multi-Camera Tracking](https://openaccess.thecvf.com/content/CVPR2024W/AICity/papers/Xie_A_Robust_Online_Multi-Camera_People_Tracking_System_With_Geometric_Consistency_CVPRW_2024_paper.pdf)
- [Deep Learning for Person Re-ID](https://viso.ai/deep-learning/deep-learning-for-person-re-identification/)
- [Aqara Spatial Intelligence CES 2026](https://www.applehomeauthority.com/aqara-redefines-the-smart-home-with-spatial-intelligence-at-ces-2026/)
- [Making Presence Detection Not Binary](https://philhawthorne.com/making-home-assistants-presence-detection-not-so-binary/)

---

## 3. Vision API Integration (GPT-4 Vision)

### Problem Statement
Getting consistent, structured output from vision APIs with proper error handling.

### Best Practices

#### 3.1 Structured Output Prompts

**Key Principle:** Write clearer specs, not longer prompts.

**Effective Prompt Template:**
```
IMPORTANT: Respond only with the following JSON structure. Do not explain.

{
  "people_detected": [
    {
      "person_id": "string",
      "location": "string (room/area)",
      "activity": "string (standing/sitting/walking/etc)",
      "confidence": float (0-1),
      "bounding_box": [x, y, width, height]
    }
  ],
  "scene_description": "string (brief)",
  "anomalies": ["string (any unusual observations)"]
}

Analyze the image and fill this structure.
```

#### 3.2 Success Criteria in Prompts

Include in every prompt:
- **Output contract:** Format, length, required sections
- **Constraints:** Scope, assumptions, what to do when uncertain
- **Success criteria:** What "done" looks like

#### 3.3 Known Limitations to Handle

| Limitation | Workaround |
|------------|------------|
| Small/rotated text | Pre-process images, rotate if needed |
| Precise object counting | Use dedicated CV models for counting |
| Non-Latin text | Use OCR preprocessing |
| Panoramic/fisheye images | Dewarp before sending |
| Spatial reasoning (chess, etc) | Use specialized models |

#### 3.4 Error Handling & Retry Strategy

**Timeout Configuration:**
- Simple completions: 10-20 seconds
- Complex/large context: 30+ seconds
- Default recommendation: 30 seconds

**Retry Strategy (Exponential Backoff):**
```python
from tenacity import retry, wait_exponential, stop_after_attempt

@retry(
    wait=wait_exponential(multiplier=1, min=1, max=60),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type((TimeoutError, RateLimitError))
)
async def analyze_frame(image: bytes, prompt: str) -> dict:
    # API call here
    pass
```

**Retry-Safe Errors:**
- 5xx server errors (retry)
- 429 rate limits (retry with backoff)
- Timeouts (retry)
- 4xx client errors (DO NOT retry)

#### 3.5 Rate Limit Headers to Monitor

| Header | Purpose |
|--------|---------|
| `x-ratelimit-limit-requests` | RPM limit |
| `x-ratelimit-remaining-requests` | Remaining requests |
| `x-ratelimit-reset-requests` | Reset time |

### Sources
- [OpenAI Error Codes](https://platform.openai.com/docs/guides/error-codes)
- [OpenAI Rate Limits Cookbook](https://developers.openai.com/cookbook/examples/how_to_handle_rate_limits/)
- [GPT-4o Vision Guide](https://getstream.io/blog/gpt-4o-vision-guide/)
- [IBM Prompt Engineering Guide 2026](https://www.ibm.com/think/prompt-engineering)
- [Azure GPT-4V Prompt Engineering](https://learn.microsoft.com/en-us/azure/ai-foundry/openai/concepts/gpt-4-v-prompt-engineering)

---

## 4. LLM Agent for Real-time Conversation

### Problem Statement
Making an LLM agent respond fast enough for natural conversation (sub-second latency).

### Best Practices

#### 4.1 Parallel Processing Architecture

**Key Insight:** LLM inference accounts for 60-70% of total latency.

**Streaming Architecture:**
```
Audio In ──▶ STT (streaming) ──▶ LLM (streaming) ──▶ TTS (streaming) ──▶ Audio Out
                  │                    │                    │
                  └────────────────────┴────────────────────┘
                           (parallel/pipelined)
```

**Implementation Pattern:**
```python
import asyncio

async def process_conversation():
    # Producer/Consumer pattern
    llm_queue = asyncio.Queue(maxsize=1)  # Backpressure

    async def llm_producer():
        async for token in stream_llm_response():
            await llm_queue.put(token)

    async def tts_consumer():
        while True:
            sentence = await collect_sentence(llm_queue)
            await synthesize_speech(sentence)

    await asyncio.gather(llm_producer(), tts_consumer())
```

#### 4.2 Dual-Model Strategy (SLM + LLM)

**Pattern:** Use fast SLM for immediate response, LLM for detailed follow-up.

```python
async def respond_with_dual_models(query: str):
    # Run both concurrently
    slm_task = asyncio.create_task(slm.respond(query))
    llm_task = asyncio.create_task(llm.respond(query))

    # Return SLM immediately, then LLM elaboration
    quick_response = await slm_task
    yield quick_response

    detailed_response = await llm_task
    yield detailed_response
```

#### 4.3 Latency Optimization Techniques

| Technique | Improvement | Notes |
|-----------|-------------|-------|
| Gemini Flash | 60% faster | Switch model |
| Prompt Caching | 80% reduction | For repeated context |
| INT8 Quantization | 3x faster | For local models |
| Connection Pooling | Variable | Reduce handshake overhead |

**Industry Benchmarks (2026):**
- Target: Sub-500ms voice-to-voice TTFB
- Achievable: ~465ms with optimized stack
- Mistral Large 2512, GPT-5.2: Sub-second first token

#### 4.4 Context Management for Speed

**Strategies:**
1. **Minimal Context:** Only include what's needed for current query
2. **Summarization:** Compress history, not raw messages
3. **Tool Results:** Summarize, don't include raw output
4. **Caching:** Cache system prompts and static context

### Sources
- [Voice AI Latency Optimization 2026](https://www.ruh.ai/blogs/voice-ai-latency-optimization)
- [Voice AI Stack 2026](https://www.assemblyai.com/blog/the-voice-ai-stack-for-building-agents)
- [Parallel SLMs and LLMs](https://webrtc.ventures/2025/06/reducing-voice-agent-latency-with-parallel-slms-and-llms/)
- [LLM Latency Benchmark 2026](https://research.aimultiple.com/llm-latency-benchmark/)

---

## 5. Event-Driven Architecture

### Problem Statement
Building a reliable, scalable event bus for IoT/camera events with proper backpressure handling.

### Best Practices

#### 5.1 MQTT + Kafka Architecture

**Key Insight:** MQTT for IoT edge, Kafka for enterprise backbone.

```
Cameras ──MQTT──▶ Edge Gateway ──Kafka──▶ Processing Services
                       │
                       ▼
                  Local Processing
                  (Frigate, CV models)
```

**Benefits:**
- MQTT: Lightweight, built for IoT, handles unreliable networks
- Kafka: Scalable, durable, supports replay

#### 5.2 Topic Structure (Unified Namespace)

**Pattern:** Hierarchical MQTT topics

```
home/
├── living_room/
│   ├── camera1/
│   │   ├── status      # online/offline/error
│   │   ├── frame       # latest frame metadata
│   │   └── events/
│   │       ├── motion
│   │       └── person
│   └── camera2/
│       └── ...
└── kitchen/
    └── camera3/
        └── ...
```

#### 5.3 Backpressure Handling

**Problem:** Producer overwhelms consumer, causing memory exhaustion.

**Solutions:**

```python
# 1. Bounded Queue
queue = asyncio.Queue(maxsize=100)

# 2. Drop oldest on overflow
async def put_with_drop(queue, item):
    if queue.full():
        try:
            queue.get_nowait()  # Drop oldest
        except asyncio.QueueEmpty:
            pass
    await queue.put(item)

# 3. Semaphore-based limiting
semaphore = asyncio.Semaphore(10)

async def process_with_limit(event):
    async with semaphore:
        await handle_event(event)
```

**Framework Recommendation:** FastStream

```python
from faststream import FastStream
from faststream.kafka import KafkaBroker

broker = KafkaBroker("localhost:9092")
app = FastStream(broker)

@broker.subscriber("camera-events")
async def handle_camera_event(event: CameraEvent):
    # Automatic backpressure handling
    await process_event(event)
```

#### 5.4 Priority Queue Implementation

**Pattern:** Redis Sorted Sets for prioritized events

```python
import redis

r = redis.Redis()

# Add event with priority (lower = higher priority)
def add_priority_event(event_id: str, priority: int, data: str):
    r.zadd("events:priority", {f"{event_id}:{data}": priority})

# Get highest priority event
def get_next_event():
    result = r.zpopmin("events:priority", count=1)
    if result:
        return result[0]
    return None
```

**Priority Levels:**
| Priority | Score | Event Type |
|----------|-------|------------|
| Critical | 0 | Person detected (unknown) |
| High | 10 | Person detected (known) |
| Medium | 50 | Motion detected |
| Low | 100 | Periodic status |

#### 5.5 Video Streaming Considerations

**Warning:** Standard pub/sub (Kafka, MQTT) falls short for real-time video due to:
- Platform-dependent design
- Rigid optimization
- Poor sub-second media handling

**Solution:** Use FrameMQ or dedicated video streaming (WebRTC, RTSP) alongside event bus.

### Sources
- [Event-Driven Architecture Patterns](https://dzone.com/articles/event-driven-architecture-real-world-iot)
- [Kafka for IoT](https://www.instaclustr.com/education/apache-kafka/kafka-for-iot-4-key-capabilities-and-top-use-cases-in-2025/)
- [FastStream](https://github.com/ag2ai/faststream)
- [Asyncio Backpressure](https://lucumr.pocoo.org/2020/1/1/async-pressure/)
- [Redis Priority Queue](https://oneuptime.com/blog/post/2026-01-21-redis-priority-queues-sorted-sets/view)

---

## 6. Database Design for Time-Series + Events

### Problem Statement
Efficiently storing and querying observations over time while supporting event-driven queries.

### Best Practices

#### 6.1 TimescaleDB on PostgreSQL

**Key Features:**
- Automatic time-based partitioning (hypertables)
- 90%+ compression typical
- Continuous aggregates (incremental materialized views)
- Full SQL compatibility

**Note:** TimescaleDB is deprecated for PostgreSQL 17. Use PostgreSQL 15 for now.

#### 6.2 Schema Design

**Observations Table (Hypertable):**
```sql
CREATE TABLE observations (
    time        TIMESTAMPTZ NOT NULL,
    camera_id   TEXT NOT NULL,
    person_id   TEXT,
    activity    TEXT,
    location    TEXT,
    confidence  FLOAT,
    metadata    JSONB,
    PRIMARY KEY (time, camera_id)
);

SELECT create_hypertable('observations', 'time');

-- Compound index for common queries
CREATE INDEX idx_observations_camera_time
ON observations (camera_id, time DESC);

CREATE INDEX idx_observations_person
ON observations (person_id, time DESC);
```

**Events Table (Hypertable):**
```sql
CREATE TABLE events (
    time        TIMESTAMPTZ NOT NULL,
    event_type  TEXT NOT NULL,
    severity    INT NOT NULL,
    source      TEXT NOT NULL,
    data        JSONB,
    PRIMARY KEY (time, event_type, source)
);

SELECT create_hypertable('events', 'time');
```

#### 6.3 Continuous Aggregates for Analytics

```sql
CREATE MATERIALIZED VIEW hourly_activity
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', time) AS bucket,
    camera_id,
    person_id,
    activity,
    COUNT(*) as count,
    AVG(confidence) as avg_confidence
FROM observations
GROUP BY bucket, camera_id, person_id, activity;

-- Refresh policy (incrementally update every hour)
SELECT add_continuous_aggregate_policy('hourly_activity',
    start_offset => INTERVAL '3 hours',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour'
);
```

#### 6.4 Retention Policy

```sql
-- Keep detailed data for 7 days
SELECT add_retention_policy('observations', INTERVAL '7 days');

-- Keep aggregates for 1 year
SELECT add_retention_policy('hourly_activity', INTERVAL '1 year');
```

#### 6.5 Query Optimization

**Use `time_bucket` instead of `date_trunc`:**
```sql
-- Good (optimized for hypertables)
SELECT time_bucket('1 hour', time) as hour, COUNT(*)
FROM observations
WHERE time > NOW() - INTERVAL '24 hours'
GROUP BY hour;

-- Avoid (slower on hypertables)
SELECT date_trunc('hour', time) as hour, COUNT(*)
FROM observations
WHERE time > NOW() - INTERVAL '24 hours'
GROUP BY hour;
```

### Sources
- [TimescaleDB GitHub](https://github.com/timescale/timescaledb)
- [TimescaleDB Best Practices](https://www.slingacademy.com/article/postgresql-with-timescaledb-best-practices-for-time-series-database-design/)
- [Tiger Data (TimescaleDB creators)](https://www.tigerdata.com/timescaledb)
- [Supabase TimescaleDB Guide](https://supabase.com/docs/guides/database/extensions/timescaledb)

---

## 7. Container Deployment

### Problem Statement
Deploying ML workloads in Docker with GPU access and efficient inter-container communication.

### Best Practices

#### 7.1 GPU Access in Docker

**Prerequisites:**
1. NVIDIA Container Toolkit installed
2. Docker configured for GPU support

**Docker Compose GPU Configuration:**
```yaml
services:
  vision-processor:
    image: vision-assistant:latest
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

**Targeting Specific GPUs:**
```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          device_ids: ['0']  # Use first GPU
          capabilities: [gpu]
```

#### 7.2 GPU Sharing Considerations

**Key Insight:** GPU access is shared at hardware/driver level, not virtualized.

| Concern | Mitigation |
|---------|------------|
| Resource contention | Limit containers per GPU |
| Memory interference | Set GPU memory limits |
| Security | Minimal container privileges |
| Multi-tenancy | Use MIG (Multi-Instance GPU) if available |

**Memory Limiting (PyTorch example):**
```python
import torch
# Limit GPU memory to 4GB
torch.cuda.set_per_process_memory_fraction(0.25)  # 25% of GPU memory
```

#### 7.3 Inter-Container Communication

**Docker Compose Networking:**
```yaml
services:
  camera-processor:
    networks:
      - vision-net

  llm-agent:
    networks:
      - vision-net

  database:
    networks:
      - vision-net

networks:
  vision-net:
    driver: bridge
```

**Service Discovery:**
- Containers reference each other by service name
- Example: `grpc://camera-processor:50051`

**Communication Patterns:**

| Pattern | Use Case | Protocol |
|---------|----------|----------|
| Request/Response | API calls | REST, gRPC |
| Streaming | Video frames | gRPC streaming, WebSocket |
| Pub/Sub | Events | MQTT, Redis Pub/Sub |
| Shared State | Caching | Redis |

#### 7.4 gRPC Between Containers

**Important:** Use container name, not `localhost` or `0.0.0.0`.

```python
# Wrong
channel = grpc.insecure_channel('localhost:50051')

# Correct
channel = grpc.insecure_channel('camera-processor:50051')
```

**With NGINX Load Balancing (for scaling):**
```yaml
services:
  nginx:
    image: nginx:latest
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - vision-processor

  vision-processor:
    deploy:
      replicas: 3
```

```nginx
# nginx.conf
upstream vision_grpc {
    server vision-processor:50051;
}

server {
    listen 50051 http2;
    location / {
        grpc_pass grpc://vision_grpc;
    }
}
```

#### 7.5 ML Container Best Practices

| Practice | Reason |
|----------|--------|
| Use NVIDIA NGC base images | Optimized for GPU workloads |
| Pin CUDA version | Reproducibility |
| Multi-stage builds | Smaller final images |
| Health checks | Detect GPU issues |
| Non-root user | Security |

**Example Dockerfile:**
```dockerfile
FROM nvcr.io/nvidia/pytorch:24.01-py3 as base

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

# Health check for GPU
HEALTHCHECK --interval=30s --timeout=10s \
  CMD python -c "import torch; assert torch.cuda.is_available()"

USER 1000
CMD ["python", "src/main.py"]
```

#### 7.6 GPU Monitoring

```bash
# Deploy DCGM Exporter for Prometheus
docker run -d --restart=unless-stopped \
  --gpus all \
  --name dcgm-exporter \
  -p 9400:9400 \
  nvcr.io/nvidia/k8s/dcgm-exporter
```

### Sources
- [Docker GPU Support](https://docs.docker.com/desktop/features/gpu/)
- [Docker Compose GPU](https://docs.docker.com/compose/how-tos/gpu-support/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
- [Docker Compose Networking](https://docs.docker.com/compose/how-tos/networking/)
- [GPU Sharing in Docker](https://agirlamonggeeks.com/can-docker-containers-share-a-gpu/)

---

## Summary: Recommended Technology Stack

| Component | Recommendation | Alternative |
|-----------|----------------|-------------|
| State Management | Home Assistant + MQTT | Custom with Redis |
| Person Tracking | YOLOv8n + DeepSORT | Frigate (if simpler needed) |
| Re-ID Model | OSNet_x1_0 | ResNet50 |
| Vision API | OpenAI GPT-4o | Local: llama.cpp + LLaVA |
| LLM Agent | Streaming + Prompt Caching | Dual SLM/LLM |
| Event Bus | MQTT (edge) + Redis (app) | Kafka (if scale needed) |
| Priority Queue | Redis Sorted Sets | Python heapq (simple) |
| Database | PostgreSQL + TimescaleDB | SQLite (development) |
| Containers | Docker Compose + NVIDIA Toolkit | Kubernetes (scale) |
| GPU Sharing | Memory limits + scheduling | NVIDIA MIG |

---

## Risk Assessment

| Area | Risk Level | Concern | Mitigation |
|------|------------|---------|------------|
| Multi-camera sync | Medium | Clock drift, network delays | NTP sync, buffer tolerance |
| Person Re-ID | High | Accuracy in varied lighting | Multiple embeddings, human verification |
| Vision API | Medium | Rate limits, costs | Caching, local fallback |
| LLM Latency | Medium | User experience | Streaming, parallel models |
| Event Backpressure | High | Memory exhaustion | Bounded queues, dropping policies |
| Database Growth | Medium | Storage costs | Retention policies, compression |
| GPU Contention | Medium | Performance degradation | Memory limits, scheduling |

---

## Next Steps

1. **Validate stack** against current llama.cpp-based local approach
2. **Design database schema** with TimescaleDB considerations
3. **Prototype event bus** with MQTT + Redis priority queue
4. **Benchmark** person re-ID models on actual camera feeds
5. **Test GPU sharing** between vision and LLM containers

---

*This research provides evidence-based recommendations. Final architecture decisions should consider the specific constraints of home deployment (limited GPU, privacy requirements, cost sensitivity).*
