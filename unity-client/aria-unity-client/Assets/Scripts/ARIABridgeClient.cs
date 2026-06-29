using System;
using System.Collections;
using System.Collections.Generic;
using System.Text;
using UnityEngine;
using NativeWebSocket;

[Serializable]
public class ARIAAnnotation
{
    public string type;
    public string text;
    public float[] mask_center_2d;  // [cx, cy] in camera pixel coords, null if no SAM2 lock
    public string detection_source; // "sam2", "yolo", "yolo_fallback", "none"
}

[Serializable]
public class ARIAResponse
{
    public string type;
    public string text;
    public ARIAAnnotation[] annotations;
    public float[] mask_center_2d;   // top-level shortcut used by renderer
    public string detection_source;
    public string target_label;
}

[Serializable]
public class ARIAHandshakeAck
{
    public string type;
    public string server;
    public string model;
    public string detector;
    public string tracker;
    public float vram_gb;
}

public class ARIABridgeClient : MonoBehaviour
{
    [Header("Connection")]
    [SerializeField] private string laptopIP   = "192.168.1.99";
    [SerializeField] private int    port       = 8765;
    [SerializeField] private float  sendIntervalSeconds = 1.5f;

    [Header("References")]
    [SerializeField] public ARIAAnnotationRenderer annotationRenderer;

    private WebSocket _ws;
    private bool      _connected  = false;
    private float     _sendTimer  = 0f;

    // Latest response — read by ARIAAnnotationRenderer
    public ARIAResponse LatestResponse { get; private set; }
    public event Action<ARIAResponse> OnResponseReceived;

    async void Start()
    {
        string uri = $"ws://{laptopIP}:{port}";
        Debug.Log($"[ARIA] Connecting to {uri}");

        _ws = new WebSocket(uri);

        _ws.OnOpen += () =>
        {
            Debug.Log("[ARIA] WebSocket OPEN");
            _connected = true;
            SendHandshake();
        };

        _ws.OnMessage += (bytes) =>
        {
            string raw = Encoding.UTF8.GetString(bytes);

            // Skip binary WAV responses
            if (bytes.Length > 4 && bytes[0] == 0x52 && bytes[1] == 0x49 &&
                bytes[2] == 0x46 && bytes[3] == 0x46)
            {
                Debug.Log("[ARIA] Received WAV audio (ignored in Stage 4 Unity client)");
                return;
            }

            HandleJsonMessage(raw);
        };

        _ws.OnError += (e) => Debug.LogError($"[ARIA] Error: {e}");
        _ws.OnClose += (e) =>
        {
            Debug.Log($"[ARIA] Closed: {e}");
            _connected = false;
        };

        await _ws.Connect();
    }

    void Update()
    {
#if !UNITY_WEBGL || UNITY_EDITOR
        if (_ws != null) _ws.DispatchMessageQueue();
#endif

        if (!_connected) return;

        _sendTimer += Time.deltaTime;
        if (_sendTimer >= sendIntervalSeconds)
        {
            _sendTimer = 0f;
            SendFrameRequest();
        }
    }

    void HandleJsonMessage(string raw)
    {
        try
        {
            // Peek at type field
            if (raw.Contains("\"handshake_ack\""))
            {
                var ack = JsonUtility.FromJson<ARIAHandshakeAck>(raw);
                Debug.Log($"[ARIA] Server: {ack.server} | detector: {ack.detector} | tracker: {ack.tracker} | VRAM: {ack.vram_gb}GB");
                return;
            }

            if (raw.Contains("\"response\""))
            {
                var response = JsonUtility.FromJson<ARIAResponse>(raw);
                LatestResponse = response;
                Debug.Log($"[ARIA] Response: '{response.text}' | source={response.detection_source} | mask={FormatCenter(response.mask_center_2d)}");
                OnResponseReceived?.Invoke(response);

                if (annotationRenderer != null)
                    annotationRenderer.HandleResponse(response);
                return;
            }

            if (raw.Contains("\"error\""))
            {
                Debug.LogWarning($"[ARIA] Server error: {raw}");
                return;
            }

            Debug.Log($"[ARIA] Unhandled message: {raw.Substring(0, Mathf.Min(80, raw.Length))}");
        }
        catch (Exception e)
        {
            Debug.LogError($"[ARIA] JSON parse error: {e.Message} | raw={raw.Substring(0, Mathf.Min(80, raw.Length))}");
        }
    }

    async void SendHandshake()
    {
        if (_ws.State != WebSocketState.Open) return;
        string msg = "{\"type\":\"handshake\",\"client\":\"aria-unity-stage4\"}";
        await _ws.SendText(msg);
        Debug.Log("[ARIA] Sent handshake");
    }

    async void SendFrameRequest()
    {
        if (_ws == null || _ws.State != WebSocketState.Open) return;

        // Send a text-only frame_text request every interval
        // In Stage 5 this will include actual camera frames
        string msg = "{\"type\":\"frame_text\",\"content\":\"What objects do you see?\",\"frame_b64\":\"\"}";
        await _ws.SendText(msg);
    }

    public async void SendTextQuery(string query)
    {
        if (_ws == null || _ws.State != WebSocketState.Open) return;
        string escaped = query.Replace("\"", "\\\"");
        string msg = $"{{\"type\":\"text\",\"content\":\"{escaped}\"}}";
        await _ws.SendText(msg);
        Debug.Log($"[ARIA] Sent query: {query}");
    }

    string FormatCenter(float[] c) => c != null && c.Length == 2 ? $"[{c[0]:F1}, {c[1]:F1}]" : "null";

    async void OnApplicationQuit()
    {
        if (_ws != null) await _ws.Close();
    }
}