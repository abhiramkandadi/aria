import asyncio, base64, io, json, logging, re, time, wave, warnings
warnings.filterwarnings('ignore')
import numpy as np, torch, websockets
from PIL import Image
from piper.voice import PiperVoice
from transformers import BitsAndBytesConfig, Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
import sys, os as _os
sys.path.insert(0, _os.path.abspath('.'))
from backend.inference.detector import ObjectDetector
from backend.inference.tracker import ObjectTracker
import backend.inference.object_map_state as obj_state

WEBSOCKET_HOST = '0.0.0.0'
WEBSOCKET_PORT = 8765
MODEL_PATH = 'C:/Users/abhir/aria/models/qwen2.5-omni-3b'
VOICE_MODEL = 'backend/tts/voices/en_US-lessac-medium.onnx'
MAX_NEW_TOKENS = 128
SAM2_WINDOW = 5

SYSTEM_PROMPT = (
    'You are ARIA, a spatially-aware AI assistant running on a Meta Quest 3. '
    'You can see the user environment through the camera. '
    'Be concise - 1 to 2 sentences max. No markdown, no bullet points. '
    'If you identify a specific object end your response with '
    'TARGET:<object_label> using the exact COCO label eg TARGET:bottle.'
)

ANNOTATION_TEMPLATE = {'type': 'label', 'text': '', 'position': {'x': 0.0, 'y': 1.5, 'z': 2.0}, 'color': '#00FF88'}

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('aria.bridge')

processor = None
model = None
tts_voice = None
detector = None
tracker = None
_model_load_time = 0.0
_tracking_state = {'target_label': None, 'frame_buffer': [], 'seed_bbox': None, 'last_mask_center': None}

def load_models():
    global processor, model, tts_voice, detector, tracker, _model_load_time
    logger.info('Loading Qwen2.5-Omni-3B (4-bit)...')
    t0 = time.perf_counter()
    processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type='nf4')
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(MODEL_PATH, quantization_config=bnb, device_map='auto', trust_remote_code=True)
    model.eval()
    _model_load_time = time.perf_counter() - t0
    logger.info(f'Qwen loaded in {_model_load_time:.1f}s | VRAM: {torch.cuda.memory_allocated(0)/1e9:.2f}GB')
    tts_voice = PiperVoice.load(VOICE_MODEL)
    logger.info('Piper TTS loaded')
    detector = ObjectDetector(device='cuda')
    logger.info('YOLO ready')
    tracker = ObjectTracker(device='cuda')
    logger.info(f'All models ready. VRAM: {torch.cuda.memory_allocated(0)/1e9:.2f}GB')

def run_inference(text, image=None):
    t0 = time.perf_counter()
    if image is not None:
        content = [{'type': 'image', 'image': image}, {'type': 'text', 'text': SYSTEM_PROMPT + chr(10) + text}]
        images = [image]
    else:
        content = [{'type': 'text', 'text': SYSTEM_PROMPT + chr(10) + text}]
        images = None
    messages = [{'role': 'user', 'content': content}]
    chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[chat_text], images=images, return_tensors='pt') if images else processor(text=[chat_text], return_tensors='pt')
    inputs = {k: v.to('cuda') if hasattr(v, 'to') else v for k, v in inputs.items()}
    t_pre = time.perf_counter() - t0
    t1 = time.perf_counter()
    with torch.no_grad():
        output_ids = model.thinker.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    t_inf = time.perf_counter() - t1
    generated = output_ids[0][inputs['input_ids'].shape[1]:]
    response = processor.decode(generated, skip_special_tokens=True).strip()
    return response, {'preprocess_s': round(t_pre,3), 'inference_s': round(t_inf,3), 'total_s': round(time.perf_counter()-t0,3)}

def extract_target_label(text):
    import re
    m = re.search(r'TARGET:(\w+)', text)
    return m.group(1).lower() if m else None

def clean_response_text(text):
    import re
    return re.sub(r'\s*TARGET:\w+', '', text).strip()

def run_detection(image):
    detections = detector.detect(image)
    obj_map = detector.build_object_map(detections)
    obj_state.update(detections, obj_map)
    return detections, obj_map

def run_tracking(image, target_label):
    state = _tracking_state
    if state['target_label'] != target_label:
        logger.info(f'[Tracker] New target: {target_label}')
        state['target_label'] = target_label
        state['frame_buffer'] = []
        state['seed_bbox'] = None
        state['last_mask_center'] = None
    state['frame_buffer'].append(image)
    if state['seed_bbox'] is None:
        dets = [d for d in obj_state._state['raw_detections'] if d['label'].lower() == target_label.lower()]
        if dets:
            state['seed_bbox'] = dets[0]['bbox_xyxy']
    if len(state['frame_buffer']) < SAM2_WINDOW or state['seed_bbox'] is None:
        return state['last_mask_center']
    try:
        results = tracker.track_sequence(state['frame_buffer'][-SAM2_WINDOW:], target_label, state['seed_bbox'])
        last = results[-1]
        if last['mask_center'] is not None:
            state['last_mask_center'] = last['mask_center']
    except Exception as e:
        logger.error(f'[Tracker] SAM2 error: {e}')
    state['frame_buffer'] = state['frame_buffer'][-2:]
    return state['last_mask_center']

def run_tts(text):
    t0 = time.perf_counter()
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        tts_voice.synthesize_wav(text, wf)
    return buf.getvalue(), time.perf_counter() - t0

async def handle_client(websocket):
    client_addr = websocket.remote_address
    logger.info('Client connected: %s', client_addr)
    try:
        async for raw_message in websocket:
            t0 = time.perf_counter()
            if isinstance(raw_message, bytes):
                try:
                    image = Image.open(io.BytesIO(raw_message)).convert('RGB')
                    detections, obj_map = run_detection(image)
                    response_text, inf_lat = run_inference('What do you see? Describe briefly.', image)
                except Exception as e:
                    logger.error('Binary frame error: %s', e)
                    response_text, inf_lat, detections, obj_map = 'Could not process frame.', {}, [], {}
                wav_bytes, tts_t = run_tts(response_text)
                await websocket.send(json.dumps({'type': 'response', 'text': response_text, 'annotations': [{**ANNOTATION_TEMPLATE, 'text': response_text}], 'detections': detections, 'object_map': obj_map, 'mask_center_2d': None, 'detection_source': 'yolo', 'latency': {**inf_lat, 'tts_s': round(tts_t,3), 'total_s': round(time.perf_counter()-t0,3)}}))
                await websocket.send(wav_bytes)
                continue
            try:
                msg = json.loads(raw_message)
            except json.JSONDecodeError:
                await websocket.send(json.dumps({'type': 'error', 'message': 'Expected JSON'}))
                continue
            msg_type = msg.get('type', 'unknown')
            if msg_type == 'handshake':
                await websocket.send(json.dumps({'type': 'handshake_ack', 'server': 'ARIA Bridge Stage 4', 'model': 'Qwen2.5-Omni-3B (4-bit)', 'detector': 'YOLOv8n', 'tracker': 'SAM2.1-hiera-tiny', 'tts': 'Piper en_US-lessac-medium', 'vram_gb': round(torch.cuda.memory_allocated(0)/1e9, 2), 'model_load_s': round(_model_load_time, 1)}))
            elif msg_type == 'text':
                content = msg.get('content', '').strip()
                if not content:
                    await websocket.send(json.dumps({'type': 'error', 'message': 'Empty content'}))
                    continue
                response_text, inf_lat = run_inference(content)
                clean_text = clean_response_text(response_text)
                wav_bytes, tts_t = run_tts(clean_text)
                await websocket.send(json.dumps({'type': 'response', 'text': clean_text, 'annotations': [{**ANNOTATION_TEMPLATE, 'text': clean_text}], 'detections': [], 'object_map': obj_state.get_object_map(), 'mask_center_2d': None, 'detection_source': 'none', 'latency': {**inf_lat, 'tts_s': round(tts_t,3), 'total_s': round(time.perf_counter()-t0,3)}}))
                await websocket.send(wav_bytes)
            elif msg_type == 'frame_text':
                content = msg.get('content', 'What do you see?').strip()
                frame_b64 = msg.get('frame_b64', '')
                target_label = msg.get('target_label', None)
                image = None
                if frame_b64:
                    try:
                        image = Image.open(io.BytesIO(base64.b64decode(frame_b64))).convert('RGB')
                    except Exception as e:
                        logger.warning('frame_b64 decode error: %s', e)
                detections, obj_map = [], {}
                if image is not None:
                    detections, obj_map = run_detection(image)
                mask_center_2d = None
                detection_source = 'yolo'
                if target_label and image is not None:
                    mask_center_2d = run_tracking(image, target_label)
                    detection_source = 'sam2' if mask_center_2d is not None else 'yolo_fallback'
                    if mask_center_2d is None:
                        mask_center_2d = obj_state.get_center_for(target_label)
                response_text, inf_lat = run_inference(content, image=image)
                new_target = extract_target_label(response_text)
                if new_target:
                    target_label = new_target
                clean_text = clean_response_text(response_text)
                wav_bytes, tts_t = run_tts(clean_text)
                total_s = time.perf_counter() - t0
                await websocket.send(json.dumps({'type': 'response', 'text': clean_text, 'annotations': [{**ANNOTATION_TEMPLATE, 'text': clean_text, 'mask_center_2d': mask_center_2d}], 'detections': detections, 'object_map': obj_map, 'mask_center_2d': mask_center_2d, 'detection_source': detection_source, 'target_label': target_label, 'latency': {**inf_lat, 'tts_s': round(tts_t,3), 'total_s': round(total_s,3)}}))
                await websocket.send(wav_bytes)
                logger.info('frame_text %.2fs | target=%s | mask=%s', total_s, target_label, mask_center_2d)
            else:
                await websocket.send(json.dumps({'type': 'error', 'message': f'Unknown type: {msg_type}'}))
    except websockets.exceptions.ConnectionClosedOK:
        logger.info('Client disconnected: %s', client_addr)
    except websockets.exceptions.ConnectionClosedError as e:
        logger.warning('Connection dropped: %s - %s', client_addr, e)
    except Exception as e:
        logger.exception('Unhandled error: %s', e)

async def main():
    logger.info('ARIA Bridge Server - Stage 4')
    load_models()
    async with websockets.serve(handle_client, WEBSOCKET_HOST, WEBSOCKET_PORT):
        logger.info('Server ready on ws://%s:%d', WEBSOCKET_HOST, WEBSOCKET_PORT)
        await asyncio.Future()

if __name__ == '__main__':
    asyncio.run(main())
